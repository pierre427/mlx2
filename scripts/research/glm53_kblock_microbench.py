"""Real-tensor microbench: packed reference versus K-block row kernels.

Uploads only the named GLM-5.3 tensors (no model), then for each projection and
one routed-expert layer compares outputs (max abs / relative error against the
reference kernel) and times both. ``per_call_ms`` is in-graph GPU time: the
call is issued ``--reps`` times inside one evaluation and divided, so fence and
launch-sync overhead is excluded. Achieved GB/s uses the packed bytes the call
must read. Run inside the owned GPU window.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL = Path("~/mlx-models/GLM-5.3-Flash-REAP50-Q3_K_M")
DENSE = [
    "blk.0.attn_q.weight",  # Q3_K 8192x4096
    "blk.0.attn_v.weight",  # Q5_K
    "blk.5.attn_v.weight",  # Q4_K
    "blk.0.attn_output.weight",  # Q4_K 4096x8192
    "blk.3.ffn_down_shexp.weight",  # Q4_K 4096x2048
    "blk.0.ssm_f_b.weight",  # Q4_0 8192x128
    "blk.3.attn_kv_a_mqa.weight",  # Q3_K 512x4096
    "output.weight",  # Q6_K 154880x4096
]
GROUPED = [("blk.3.attn_k_b.weight", 64), ("blk.3.attn_v_b.weight", 64)]
MOE = "blk.3"


def timed(fn, reps, mx):
    mx.eval(fn())
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(reps)])
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def error(a, b, mx):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return {"max_abs": float(mx.max(mx.abs(a - b)).item()),
            "rel_l2": float((mx.linalg.norm(a - b) / mx.linalg.norm(a)).item())}


VARIANTS = {"v0": {"vec": 0}, "v1": {"vec": 1}, "v2": {"vec": 2}, "v4": {"vec": 3}, "v5": {"vec": 4},
            "exact": {"vec": 1, "exact": 1},
            "r1": {"vec": 1, "rows": 1}, "r2": {"vec": 1, "rows": 2}, "r8": {"vec": 1, "rows": 8}}


def prefill_tiles(args, weights, mx, rows):
    """nax_prefill against kblock_nax_prefill(MT): bitwise and TF/s at prompt width."""
    import numpy as np

    from mlx2.runtime.gguf.kblock_prefill_metal import kblock_nax_prefill
    from mlx2.runtime.gguf.nax_metal import nax_prefill

    names = ["blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.5.attn_v.weight",
             "blk.0.attn_output.weight", "blk.3.ffn_up_shexp.weight", "blk.3.ffn_down_shexp.weight",
             "blk.3.attn_q_b.weight"]
    m = args.prefill
    for name in names:
        w = weights.packed(name)
        x = mx.random.normal((m, w.shape[1])).astype(mx.float32)
        flops = 2.0 * m * w.shape[0] * w.shape[1]
        ref = nax_prefill(x, w)
        mx.eval(ref)
        t_ref = min(_timed_once(lambda w=w, x=x: [nax_prefill(x, w) for _ in range(3)], mx) for _ in range(3)) / 3
        ref_np = np.array(ref)
        line = f"prefill {name:30s} t{w.ggml_type:<3d} m={m} nax {t_ref*1e3:7.2f} ms {flops/t_ref/1e12:5.2f} TF/s"
        for mt in (1, 2, 4):
            out = kblock_nax_prefill(x, w, mt=mt)
            mx.eval(out)
            # time includes the one-off half cast, as served
            t = min(_timed_once(lambda w=w, x=x, mt=mt: [kblock_nax_prefill(x, w, mt=mt) for _ in range(3)], mx)
                    for _ in range(3)) / 3
            mism = int(np.count_nonzero(np.array(out).view(np.int32) != ref_np.view(np.int32)))
            rows.append({"label": f"prefill {name} MT{mt}", "ggml_type": w.ggml_type, "rows": m,
                         "nax_ms": t_ref * 1e3, "kblock_ms": t * 1e3, "kblock_tflop_s": flops / t / 1e12,
                         "nax_tflop_s": flops / t_ref / 1e12, "bitwise_mismatches": mism, "outputs": int(out.size)})
            line += f" | MT{mt} {t*1e3:7.2f} ms {flops/t/1e12:5.2f} TF/s mism {mism}"
        # Stage diagnostics at MT=2, interleaved with a full run in the same process.
        for diag, what in ((0, "full"), (1, "no-decode"), (2, "no-mma"), (0, "full")):
            mx.eval(kblock_nax_prefill(x, w, mt=2, diag=diag))
            t = min(_timed_once(lambda w=w, x=x, d=diag: [kblock_nax_prefill(x, w, mt=2, diag=d) for _ in range(3)], mx)
                    for _ in range(3)) / 3
            rows.append({"label": f"prefill diag {name} {what}", "ms": t * 1e3})
            line += f" || {what} {t*1e3:6.2f}"
        print(line, flush=True)
    # Prefill-width cost of the tile-ineligible Q4_0 KDA low-rank tensors (K=128).
    from mlx2.runtime.gguf.kblock_metal import kblock_dense

    for name in ("blk.0.ssm_f_b.weight", "blk.0.ssm_g_b.weight", "blk.0.ssm_f_a.weight"):
        w = weights.packed(name)
        x = mx.random.normal((m, w.shape[1])).astype(mx.float32)
        mx.eval(kblock_dense(x, w))
        t = min(_timed_once(lambda w=w, x=x: [kblock_dense(x, w) for _ in range(3)], mx) for _ in range(3)) / 3
        flops = 2.0 * m * w.shape[0] * w.shape[1]
        mism = int(np.count_nonzero(np.array(kblock_dense(x, w)).view(np.int32)
                                    != np.array(w(x)).view(np.int32)))
        rows.append({"label": f"prefill rows-kernel {name}", "rows": m, "kblock_ms": t * 1e3,
                     "kblock_tflop_s": flops / t / 1e12, "bitwise_mismatches_vs_packed": mism})
        print(f"prefill rows-kernel {name:30s} t{w.ggml_type:<3d} m={m} {t*1e3:7.2f} ms "
              f"{flops/t/1e12:5.2f} TF/s (x34 layers = {34*t*1e3:6.1f} ms) mism vs packed {mism}", flush=True)


def prefill_v2(args, weights, mx, rows):
    """Tile v2 against the current tile: TF/s, bitwise, and fp64-RMS class evidence."""
    import numpy as np

    from mlx2.runtime.gguf.k_blocks import decode_k_blocks
    from mlx2.runtime.gguf.kblock_prefill_metal import (
        kblock_nax_prefill,
        kblock_nax_prefill_v2,
    )

    m = args.prefill
    rng = np.random.default_rng(13)
    for name in ["blk.0.attn_q.weight", "blk.5.attn_v.weight", "blk.0.attn_output.weight",
                 "blk.3.ffn_up_shexp.weight", "blk.3.attn_q_b.weight"]:
        w = weights.packed(name)
        xs = rng.standard_normal((m, w.shape[1])).astype(np.float32)
        x = mx.array(xs)
        flops = 2.0 * m * w.shape[0] * w.shape[1]
        cur = lambda w=w, x=x: kblock_nax_prefill(x, w, mt=2)
        new = lambda w=w, x=x: kblock_nax_prefill_v2(x, w)
        # "reg" slot: the v2 tile without load prefetch (prefetch=0) for an A/B.
        reg = lambda w=w, x=x: kblock_nax_prefill_v2(x, w, prefetch=0)
        times = {}
        for label, fn in (("cur", cur), ("v2", new), ("reg", reg), ("cur", cur), ("v2", new), ("reg", reg)):
            mx.eval(fn())
            t = min(_timed_once(lambda fn=fn: [fn() for _ in range(3)], mx) for _ in range(3)) / 3
            times[label] = min(times.get(label, 1e9), t)
        yc, yn, yr = np.array(cur()), np.array(new()), np.array(reg())
        w64 = decode_k_blocks(np.array(w.data).tobytes(), w.ggml_type).astype(np.float64).reshape(w.shape)
        y64 = xs.astype(np.float64) @ w64.T
        rc = float(np.sqrt(np.mean((yc - y64) ** 2)))
        rn = float(np.sqrt(np.mean((yn - y64) ** 2)))
        mism = int(np.count_nonzero(yc.view(np.int32) != yn.view(np.int32)))
        rr = float(np.sqrt(np.mean((yr - y64) ** 2)))
        row = {"label": f"prefill v2 {name}", "rows": m, "cur_ms": times["cur"] * 1e3, "v2_ms": times["v2"] * 1e3,
               "cur_tflop_s": flops / times["cur"] / 1e12, "v2_tflop_s": flops / times["v2"] / 1e12,
               "bitwise_mismatches_vs_cur": mism, "outputs": int(yc.size),
               "fp64_rms_cur": rc, "fp64_rms_v2": rn, "fp64_ratio_v2_over_cur": rn / rc,
               "finite": bool(np.isfinite(yn).all()),
               "reg_ms": times["reg"] * 1e3, "reg_tflop_s": flops / times["reg"] / 1e12,
               "fp64_ratio_reg_over_cur": rr / rc, "reg_finite": bool(np.isfinite(yr).all())}
        rows.append(row)
        print(f"prefill v2 {name:30s} t{w.ggml_type:<3d} cur {row['cur_ms']:6.2f} ms {row['cur_tflop_s']:5.1f} TF/s"
              f" | v2 {row['v2_ms']:6.2f} ms {row['v2_tflop_s']:5.1f} TF/s | mism {mism}/{yc.size}"
              f" fp64 ratio v2/cur {row['fp64_ratio_v2_over_cur']:.4f} finite {row['finite']}"
              f" | v2 no-prefetch {row['reg_ms']:6.2f} ms {row['reg_tflop_s']:5.1f} TF/s | prefetch vs no-prefetch "
              f"bitwise mismatches {int(np.count_nonzero(yn.view(np.int32) != yr.view(np.int32)))}", flush=True)


def prefill_scratch(args, weights, mx, rows):
    """Decode-once dense prefill (v2d) against the v2 tile: time split, bits, TF/s."""
    import numpy as np

    from mlx2.runtime.gguf.kblock_prefill_metal import (
        kblock_decode_fp16,
        kblock_nax_prefill_v2,
        kblock_nax_prefill_v2d,
    )

    m = args.prefill
    rng = np.random.default_rng(13)
    for name in ["blk.0.attn_q.weight", "blk.5.attn_v.weight", "blk.0.attn_output.weight",
                 "blk.3.ffn_up_shexp.weight", "blk.3.attn_q_b.weight"]:
        w = weights.packed(name)
        x = mx.array(rng.standard_normal((m, w.shape[1])).astype(np.float32))
        x16 = x.astype(mx.float16)
        wd = kblock_decode_fp16(w)
        mx.eval(x16, wd)
        flops = 2.0 * m * w.shape[0] * w.shape[1]
        arms = {"v2": lambda w=w, x=x: kblock_nax_prefill_v2(x, w),
                "v2d": lambda w=w, x=x: kblock_nax_prefill_v2d(x, w),
                "decode": lambda w=w: kblock_decode_fp16(w),
                "mm64": lambda w=w, x=x, wd=wd: kblock_nax_prefill_v2d(x, w, decoded=wd),
                "mm64x128": lambda w=w, x=x, wd=wd: kblock_nax_prefill_v2d(x, w, decoded=wd, tn=128),
                "mm128": lambda w=w, x=x, wd=wd: kblock_nax_prefill_v2d(x, w, decoded=wd, tm=128, tn=128, sg=8),
                "mx.matmul": lambda x16=x16, wd=wd: x16 @ wd.T}
        times = {}
        for _ in range(2):
            for label, fn in arms.items():
                mx.eval(fn())
                t = min(_timed_once(lambda fn=fn: [fn() for _ in range(3)], mx) for _ in range(3)) / 3
                times[label] = min(times.get(label, 1e9), t)
        y2 = np.array(arms["v2"]())
        mism = {label: int(np.count_nonzero(y2.view(np.int32) != np.array(arms[label]()).view(np.int32)))
                for label in ("v2d", "mm64x128", "mm128")}
        row = {"label": f"prefill scratch {name}", "rows": m, "ggml_type": w.ggml_type, "shape": list(w.shape),
               **{f"{k_}_ms": v * 1e3 for k_, v in times.items()},
               **{f"{k_}_tflop_s": flops / v / 1e12 for k_, v in times.items() if k_ != "decode"},
               **{f"bitwise_mismatches_{k_}_vs_v2": v for k_, v in mism.items()}, "outputs": int(y2.size)}
        rows.append(row)
        print(f"scratch {name:28s} t{w.ggml_type:<3d} {w.shape[0]}x{w.shape[1]} | v2 {times['v2']*1e3:6.3f} ms "
              f"{flops/times['v2']/1e12:5.1f} TF/s | v2d {times['v2d']*1e3:6.3f} ms {flops/times['v2d']/1e12:5.1f} "
              f"(decode {times['decode']*1e3:6.3f} + mm {times['mm64']*1e3:6.3f}) | mm 64x128 "
              f"{times['mm64x128']*1e3:6.3f} 128x128 {times['mm128']*1e3:6.3f} | mx.matmul {times['mx.matmul']*1e3:6.3f}"
              f" | mism v2d {mism['v2d']} 64x128 {mism['mm64x128']} 128 {mism['mm128']} /{y2.size}", flush=True)


def tok_share(args, weights, kblock_metal, mx, rows):
    """Cold W-row decode projections: one token per grid slice vs token-shared chunk law."""
    import numpy as np

    linear = [i for i in range(45) if f"blk.{i}.ssm_beta.weight" in weights.tensors]
    mla = [i for i in range(45) if i not in linear]
    kinds = {"attn_q Q3_K (KDA x34)": [f"blk.{i}.attn_q.weight" for i in linear],
             "attn_v (KDA x34)": [f"blk.{i}.attn_v.weight" for i in linear],
             "attn_output (KDA x34)": [f"blk.{i}.attn_output.weight" for i in linear],
             "ffn_up_shexp (x42)": [f"blk.{i}.ffn_up_shexp.weight" for i in range(3, 45)],
             "attn_q_b (MLA x11)": [f"blk.{i}.attn_q_b.weight" for i in mla],
             "output Q6_K (x1)": ["output.weight"]}
    for label, names in kinds.items():
        ws = [weights.packed(n) for n in names]
        if not all(w.ggml_type in (11, 12, 13, 14) for w in ws):
            print(f"tok {label}: skipped (formats {sorted({w.ggml_type for w in ws})})", flush=True)
            continue
        nbytes = sum(w.tensor.nbytes for w in ws)
        for width in args.tok_widths:
            x = mx.random.normal((width, ws[0].shape[1])).astype(mx.float32)
            mx.eval(x)
            arms = {"slices": lambda ws=ws, x=x: [kblock_metal.kblock_dense(x, w, tok=1) for w in ws],
                    "shared": lambda ws=ws, x=x, width=width: [kblock_metal.kblock_dense(x, w, tok=min(width, 8))
                                                                for w in ws],
                    "shared2": lambda ws=ws, x=x, width=width: [kblock_metal.kblock_dense(x, w, tok=min(width, 2))
                                                                 for w in ws],
                    "shared4": lambda ws=ws, x=x, width=width: [kblock_metal.kblock_dense(x, w, tok=min(width, 4))
                                                                 for w in ws]}
            times = {}
            for _ in range(2):
                for name, fn in arms.items():
                    mx.eval(fn())
                    t = min(_timed_once(fn, mx) for _ in range(5))
                    times[name] = min(times.get(name, 1e9), t)
            a = np.array(kblock_metal.kblock_dense(x, ws[0], tok=1))
            b = np.array(kblock_metal.kblock_dense(x, ws[0], tok=min(width, 8)))
            one = [np.array(kblock_metal.kblock_dense(x[i:i + 1], ws[0], tok=1)) for i in range(width)]
            mism = int(np.count_nonzero(a.view(np.int32) != b.view(np.int32)))
            mism_one = int(np.count_nonzero(np.concatenate(one).view(np.int32) != b.view(np.int32)))
            row = {"label": f"tok {label}", "width": width, "slices_ms_per_call": times["slices"] * 1e3 / len(ws),
                   "shared_ms_per_call": times["shared"] * 1e3 / len(ws),
                   "speedup": times["slices"] / times["shared"],
                   "shared2_ms_per_call": times["shared2"] * 1e3 / len(ws),
                   "shared4_ms_per_call": times["shared4"] * 1e3 / len(ws),
                   "shared_weight_gb_s": nbytes / times["shared"] / 1e9,
                   "bitwise_mismatches_vs_slices": mism, "bitwise_mismatches_vs_one_row_calls": mism_one,
                   "outputs": int(a.size)}
            rows.append(row)
            print(f"tok {label:24s} W{width:<2d} slices {row['slices_ms_per_call']:.4f} ms | shared "
                  f"{row['shared_ms_per_call']:.4f} ms x{row['speedup']:.2f} ({row['shared_weight_gb_s']:.0f} GB/s weights)"
                  f" | tok2 {row['shared2_ms_per_call']:.4f} tok4 {row['shared4_ms_per_call']:.4f}"
                  f" | mism vs slices {mism} vs 1-row calls {mism_one} /{a.size}", flush=True)


def lowrank(args, weights, kblock_metal, mx, rows):
    """Cold KDA low-rank second stage (ssm_f_b + ssm_g_b, K=128) as decode runs it."""
    import numpy as np

    linear = [i for i in range(45) if f"blk.{i}.ssm_beta.weight" in weights.tensors]
    pairs = [(weights.packed(f"blk.{i}.ssm_f_b.weight"), weights.packed(f"blk.{i}.ssm_g_b.weight")) for i in linear]
    k = pairs[0][0].shape[1]
    xs = [mx.random.normal((1, k)).astype(mx.float32) for _ in range(2)]
    mx.eval(xs)
    nbytes = sum(a.tensor.nbytes + b.tensor.nbytes for a, b in pairs)
    print(f"lowrank formats {pairs[0][0].ggml_type}/{pairs[0][1].ggml_type} shape {pairs[0][0].shape}", flush=True)
    ref = np.array(kblock_metal.kblock_multi(xs, list(pairs[0]), rows=4))
    for r in (1, 2, 4, 8):
        fn = lambda r=r: [kblock_metal.kblock_multi(xs, [a, b], rows=r) for a, b in pairs]
        mx.eval(fn())
        t = min(_timed_once(fn, mx) for _ in range(7))
        out = np.array(kblock_metal.kblock_multi(xs, list(pairs[0]), rows=r))
        mism = int(np.count_nonzero(out.view(np.int32) != ref.view(np.int32)))
        rows.append({"label": f"lowrank multi rows={r}", "us_per_layer": t * 1e6 / len(pairs),
                     "gb_s": nbytes / t / 1e9, "mismatches_vs_rows4": mism})
        print(f"lowrank multi rows={r}  {t*1e6/len(pairs):7.2f} us/layer {nbytes/t/1e9:6.1f} GB/s  mism {mism}", flush=True)
    fn = lambda: [mx.zeros((1, 16384)) + 0 for _ in pairs]
    mx.eval(fn())
    t = min(_timed_once(fn, mx) for _ in range(7))
    print(f"lowrank empty-launch floor {t*1e6/len(pairs):7.2f} us/layer", flush=True)


def hc_decode(args, weights, mx, rows):
    """One decode token's HC work across 45 layers in one graph (M=1, fused_fp32)."""
    from mlx2.runtime.models.glm53_gguf import hc_collapse, hc_expand, rms

    h = mx.random.normal((1, 1, 4, 4096)).astype(mx.float32)
    branch = mx.random.normal((1, 1, 4096)).astype(mx.float32)
    mx.eval(h, branch)
    eps = 1e-5

    def full():
        x = h
        outs = []
        for i in range(45):
            p = f"blk.{i}"
            for half, norm in ((".hc_attn", ".attn_norm.weight"), (".hc_ffn", ".ffn_norm.weight")):
                c, post, comb = hc_collapse(x, weights, p + half, eps, fused=True)
                outs.append(rms(c, weights.array(p + norm), eps))
                x = hc_expand(branch, x, post, comb, fused=True)
        return [x] + outs

    def mixes_only():
        outs = []
        for i in range(45):
            for half in (".hc_attn", ".hc_ffn"):
                z = rms(h.reshape(1, 1, -1), None, eps)
                outs.append(weights.linear(f"blk.{i}" + half + "_fn")(z))
        return outs

    for label, fn in (("hc full (collapse+norm+expand) x90", full), ("hc mixes rms+matmul only x90", mixes_only)):
        mx.eval(fn())
        t = min(_timed_once(fn, mx) for _ in range(7))
        rows.append({"label": label, "ms_per_token": t * 1e3})
        print(f"{label:40s} {t*1e3:7.3f} ms/token", flush=True)


def moe_union(args, weights, kblock_metal, mx, rows):
    """Multi-row routed MoE: per-slot kblock_moe vs the expert-union kernel."""
    import numpy as np

    rng = np.random.default_rng(23)
    layers = list(range(3, 45))
    trip = [tuple(weights.packed(f"blk.{i}.ffn_{p}_exps.weight") for p in ("gate", "up", "down")) for i in layers]
    print("moe union formats", [t.ggml_type for t in trip[0]], trip[0][0].shape, flush=True)
    for width in args.tok_widths:
        for kind in ("random", "correlated"):
            ids_list = []
            for _ in layers:
                if kind == "random":
                    ids = np.stack([rng.choice(144, 8, replace=False) for _ in range(width)])
                else:  # tree rows: each row keeps ~6 of the previous row's experts
                    row = rng.choice(144, 8, replace=False)
                    out = []
                    for _ in range(width):
                        keep = rng.choice(8, 6, replace=False)
                        fresh = rng.choice(np.setdiff1d(np.arange(144), row), 2, replace=False)
                        row = np.concatenate([row[keep], fresh])
                        out.append(row)
                    ids = np.stack(out)
                ids_list.append(mx.array(ids.astype(np.uint32)))
            x = mx.random.normal((width, trip[0][0].shape[2])).astype(mx.float32)
            scores = mx.random.uniform(shape=(width, 8)).astype(mx.float32)
            mx.eval(x, scores, ids_list)
            uniq = float(np.mean([len(np.unique(np.array(i))) for i in ids_list]))
            arms = {"slots": lambda x=x, ids_list=ids_list, scores=scores: [
                        kblock_metal.kblock_moe(x, ids, scores, *t) for ids, t in zip(ids_list, trip)],
                    "union": lambda x=x, ids_list=ids_list, scores=scores: [
                        kblock_metal.kblock_moe_union(x, ids, scores, *t) for ids, t in zip(ids_list, trip)],
                    "union_g1": lambda x=x, ids_list=ids_list, scores=scores: [
                        kblock_metal.kblock_moe_union(x, ids, scores, *t, group=1) for ids, t in zip(ids_list, trip)],
                    "union_g2": lambda x=x, ids_list=ids_list, scores=scores: [
                        kblock_metal.kblock_moe_union(x, ids, scores, *t, group=2) for ids, t in zip(ids_list, trip)]}
            times = {}
            for _ in range(2):
                for name, fn in arms.items():
                    mx.eval(fn())
                    t = min(_timed_once(fn, mx) for _ in range(5))
                    times[name] = min(times.get(name, 1e9), t)
            a = np.array(kblock_metal.kblock_moe(x, ids_list[0], scores, *trip[0]))
            b = np.array(kblock_metal.kblock_moe_union(x, ids_list[0], scores, *trip[0]))
            mism = int(np.count_nonzero(a.view(np.int32) != b.view(np.int32)))
            row = {"label": f"moe union W{width} {kind}", "unique_experts": uniq,
                   "slots_ms_per_layer": times["slots"] * 1e3 / len(layers),
                   "union_ms_per_layer": times["union"] * 1e3 / len(layers),
                   "speedup": times["slots"] / times["union"], "bitwise_mismatches": mism, "outputs": int(a.size)}
            rows.append(row)
            print(f"moe union W{width:<2d} {kind:10s} uniq {uniq:5.1f} | slots {row['slots_ms_per_layer']:.4f} ms | union "
                  f"{row['union_ms_per_layer']:.4f} ms x{row['speedup']:.2f} | g1 {times['union_g1']*1e3/len(layers):.4f} "
                  f"g2 {times['union_g2']*1e3/len(layers):.4f} | mism {mism}/{a.size}", flush=True)


def moe_v2(args, weights, mx, rows):
    """Segmented expert prefill v2 against the current kernel: time, bits, fp64 class."""
    import numpy as np

    from mlx2.runtime.gguf.k_blocks import decode_k_blocks
    from mlx2.runtime.gguf.kblock_prefill_metal import (
        kblock_segmented_moe_prefill_v3,
        kblock_segmented_moe_prefill_w8a8,
    )
    from mlx2.runtime.gguf.segmented_prefill import segmented_moe_prefill

    m = args.prefill
    rng = np.random.default_rng(17)
    gate, up, down = (weights.packed(f"blk.3.ffn_{p}_exps.weight") for p in ("gate", "up", "down"))
    experts, hidden, k = gate.shape
    xs = rng.standard_normal((m, k)).astype(np.float32)
    x = mx.array(xs)
    scores_np = rng.uniform(size=(m, 8)).astype(np.float32)
    scores = mx.array(scores_np)
    for label, pool in (("all-144", experts), ("subset-8", 8)):
        if pool == experts:
            ids_np = np.stack([rng.choice(experts, 8, replace=False) for _ in range(m)]).astype(np.uint32)
        else:
            subset = rng.choice(experts, 8, replace=False)
            ids_np = np.stack([rng.permutation(subset) for _ in range(m)]).astype(np.uint32)
        ids = mx.array(ids_np)
        cur = lambda ids=ids: segmented_moe_prefill(x, ids, scores, gate, up, down, mt=2, half_x=True)
        new = lambda ids=ids: kblock_segmented_moe_prefill_v3(x, ids, scores, gate, up, down)
        times = {}
        for name, fn in (("cur", cur), ("v2", new), ("cur", cur), ("v2", new)):
            mx.eval(fn())
            t = min(_timed_once(lambda fn=fn: [fn() for _ in range(2)], mx) for _ in range(3)) / 2
            times[name] = min(times.get(name, 1e9), t)
        yc, yn = np.array(cur()), np.array(new())
        flops = 2.0 * m * 8 * 3 * hidden * k
        row = {"label": f"moe v2 {label}", "rows": m, "cur_ms": times["cur"] * 1e3, "v2_ms": times["v2"] * 1e3,
               "cur_tflop_s": flops / times["cur"] / 1e12, "v2_tflop_s": flops / times["v2"] / 1e12,
               "bitwise_mismatches_vs_cur": int(np.count_nonzero(yc.view(np.int32) != yn.view(np.int32))),
               "outputs": int(yc.size), "finite": bool(np.isfinite(yn).all())}
        if pool == 8:
            def dense(wm, e):
                rb = wm.row_bytes * wm.shape[1]
                raw = np.array(wm.data[e * rb:(e + 1) * rb]).tobytes()
                return decode_k_blocks(raw, wm.ggml_type).astype(np.float64).reshape(wm.shape[1], wm.shape[2])
            y64 = np.zeros((m, k))
            x64 = xs.astype(np.float64)
            for e in sorted({int(v) for v in ids_np.reshape(-1)}):
                g = np.minimum(x64 @ dense(gate, e).T, 10.0)
                u = np.clip(x64 @ dense(up, e).T, -10.0, 10.0)
                h = (g / (1.0 + np.exp(-g))) * u
                out = h @ dense(down, e).T
                weight = np.where(ids_np == e, scores_np, 0.0).sum(-1, keepdims=True)
                y64 += weight * out
            rc = float(np.sqrt(np.mean((yc - y64) ** 2)))
            rn = float(np.sqrt(np.mean((yn - y64) ** 2)))
            yw = np.array(kblock_segmented_moe_prefill_w8a8(x, ids, scores, gate, up, down))
            rw = float(np.sqrt(np.mean((yw - y64) ** 2)))
            row.update({"fp64_rms_cur": rc, "fp64_rms_v2": rn, "fp64_ratio_v2_over_cur": rn / rc,
                        "fp64_rms_w8a8": rw, "fp64_ratio_w8a8_over_cur": rw / rc,
                        "w8a8_rel_l2_vs_fp64": float(np.linalg.norm(yw - y64) / np.linalg.norm(y64)),
                        "cur_rel_l2_vs_fp64": float(np.linalg.norm(yc - y64) / np.linalg.norm(y64)),
                        "w8a8_finite": bool(np.isfinite(yw).all())})
            print(f"moe w8a8 subset-8 fp64: rel L2 cur {row['cur_rel_l2_vs_fp64']:.3e} w8a8 "
                  f"{row['w8a8_rel_l2_vs_fp64']:.3e} ratio {row['fp64_ratio_w8a8_over_cur']:.2f} "
                  f"finite {row['w8a8_finite']}", flush=True)
        if pool == experts:
            from mlx2.runtime.gguf.kblock_prefill_metal import segmented_gate_up_v2_only

            diag_line = ""
            for diag, what in ((0, "gate_up full"), (1, "no-decode"), (2, "no-mma"), (3, "no-gather"), (0, "full")):
                fn = lambda d=diag, ids=ids: segmented_gate_up_v2_only(x, ids, gate, up, diag=d)
                mx.eval(fn())
                t = min(_timed_once(lambda fn=fn: [fn() for _ in range(2)], mx) for _ in range(3)) / 2
                row[f"diag_{what}_ms"] = t * 1e3
                diag_line += f" | {what} {t*1e3:6.2f}"
            print(f"moe v2 diag m={m}{diag_line}", flush=True)
        rows.append(row)
        print(f"moe v2 {label:9s} m={m} cur {row['cur_ms']:7.2f} ms {row['cur_tflop_s']:5.1f} TF/s | v2 "
              f"{row['v2_ms']:7.2f} ms {row['v2_tflop_s']:5.1f} TF/s | mism {row['bitwise_mismatches_vs_cur']}/"
              f"{row['outputs']} finite {row['finite']}"
              + (f" fp64 ratio v2/cur {row['fp64_ratio_v2_over_cur']:.4f}" if "fp64_ratio_v2_over_cur" in row else ""),
              flush=True)


def moe_tune(args, weights, mx, rows):
    """MoE prefill tile tuning on captured real routing: stages and TM sweep."""
    import numpy as np

    from mlx2.runtime.gguf import segmented_prefill as seg
    from mlx2.runtime.gguf.kblock_prefill_metal import (
        _segmented_gate_up_v3,
        _segmented_kernel_v2,
        kblock_segmented_moe_prefill_v3,
        kblock_segmented_moe_prefill_v4,
        kblock_segmented_moe_prefill_w8a8,
    )

    routing = np.load(args.routing)
    layers = sorted({k.split(".")[1] for k in routing.files}, key=int)
    picks = [layers[0], layers[len(layers) // 2], layers[-1]]
    rng = np.random.default_rng(23)

    def timed(fn, reps=3):
        mx.eval(fn())
        return min(_timed_once(lambda: [fn() for _ in range(reps)], mx) for _ in range(3)) / reps

    def tune_layer(layer):
        prefix = f"blk.{layer}"
        ids = mx.array(routing[f"{prefix}.ids"].astype(np.uint32))
        scores = mx.array(routing[f"{prefix}.scores"].astype(np.float32))
        gate, up, down = (weights.packed(f"{prefix}.ffn_{p}_exps.weight") for p in ("gate", "up", "down"))
        experts, hidden, k = gate.shape
        m = ids.shape[0]
        x = mx.array(rng.standard_normal((m, k)).astype(np.float32))
        counts = np.bincount(np.array(ids).reshape(-1), minlength=experts)
        flops = 2.0 * m * 8 * 3 * hidden * k
        row = {"label": f"moe tune {prefix}", "rows": m, "active_experts": int((counts > 0).sum()),
               "rows_per_expert_max": int(counts.max()), "rows_per_expert_median": float(np.median(counts[counts > 0]))}
        cur = lambda: seg.segmented_moe_prefill(x, ids, scores, gate, up, down, mt=2, half_x=True)
        variants = [("cur", cur)] + [(f"v3 tm{t}", lambda t=t: kblock_segmented_moe_prefill_v3(
            x, ids, scores, gate, up, down, tm=t)) for t in (16, 32, 64)] + [
            ("v3 tm32 gu-tn32", lambda: kblock_segmented_moe_prefill_v3(
                x, ids, scores, gate, up, down, tm=32, gate_up_tn=32)),
            ("v3 tm32 gate_up-no-decode", lambda: kblock_segmented_moe_prefill_v3(
                x, ids, scores, gate, up, down, tm=32, diag=1)),
            ("v4 shared-decode", lambda: kblock_segmented_moe_prefill_v4(x, ids, scores, gate, up, down)),
            ("v4 gate_up-only", lambda: kblock_segmented_moe_prefill_v4(
                x, ids, scores, gate, up, down, share_down=False)),
            ("w8a8 gate_up", lambda: kblock_segmented_moe_prefill_w8a8(x, ids, scores, gate, up, down)),
            ("v3 tm32 no-prefetch", lambda: kblock_segmented_moe_prefill_v3(
                x, ids, scores, gate, up, down, tm=32, prefetch=0))]
        for _ in range(2):  # interleaved twice, best kept
            for name, fn in variants:
                row[name + "_ms"] = min(row.get(name + "_ms", 1e9), timed(fn) * 1e3)
        v3_out = np.array(kblock_segmented_moe_prefill_v3(x, ids, scores, gate, up, down, tm=32))
        v4_out = np.array(kblock_segmented_moe_prefill_v4(x, ids, scores, gate, up, down))
        v3_nopf = np.array(kblock_segmented_moe_prefill_v3(x, ids, scores, gate, up, down, tm=32, prefetch=0))
        print(f"   v3 prefetch vs no-prefetch bitwise mismatches "
              f"{int(np.count_nonzero(v3_out.view(np.int32) != v3_nopf.view(np.int32)))}/{v3_out.size}", flush=True)
        row["v4_bitwise_mismatches_vs_v3"] = int(np.count_nonzero(v3_out.view(np.int32) != v4_out.view(np.int32)))
        row["v4_finite"] = bool(np.isfinite(v4_out).all())
        print(f"   v4 vs v3 tm32 bitwise mismatches {row['v4_bitwise_mismatches_vs_v3']}/{v3_out.size} "
              f"finite {row['v4_finite']}", flush=True)
        for t in (16, 32, 64):
            plan = seg.build_plan(ids, experts, t)
            mx.eval(plan.tile_count)
            row[f"tiles_tm{t}"] = int(np.array(plan.tile_count)[0])
        # Stage breakdown at TM=32.
        stages = {}
        mx.eval(kblock_segmented_moe_prefill_v3(x, ids, scores, gate, up, down, tm=32, stages=stages))
        plan, xs, mid = stages["plan"], stages["xs"], stages["mid"]
        row["stage_plan_ms"] = timed(lambda: seg.build_plan(ids, experts, 32).tiles) * 1e3
        row["stage_gather_ms"] = timed(lambda: mx.take(x.astype(mx.float16), plan.order // 8, axis=0)) * 1e3
        row["stage_gate_up_ms"] = timed(lambda: _segmented_gate_up_v3()(
            inputs=[xs, plan.tiles, plan.tile_count, gate.data, up.data],
            template=[("K", k), ("N", hidden), ("ROW_BYTES", gate.row_bytes), ("FORMAT", gate.ggml_type), ("TMT", 32),
                      ("TNT", 64), ("DIAG", 0), ("PREFETCH", 1)],
            grid=(((hidden + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
            output_shapes=[(plan.assignment_rows, hidden)], output_dtypes=[mx.float32])[0]) * 1e3
        mid16 = mid.astype(mx.float16)
        row["stage_down_ms"] = timed(lambda: _segmented_kernel_v2("down")(
            inputs=[mid16, plan.tiles, plan.tile_count, down.data],
            template=[("K", hidden), ("N", k), ("ROW_BYTES", down.row_bytes), ("FORMAT", down.ggml_type), ("TMT", 32)],
            grid=(((k + 63) // 64) * 128, plan.max_tiles, 1), threadgroup=(128, 1, 1),
            output_shapes=[(plan.assignment_rows, k)], output_dtypes=[mx.float32])[0]) * 1e3
        rows.append(row)
        print(f"moe tune {prefix:7s} active {row['active_experts']} rows/expert median {row['rows_per_expert_median']:.0f}"
              f" max {row['rows_per_expert_max']} | tiles tm16/32/64 {row['tiles_tm16']}/{row['tiles_tm32']}/{row['tiles_tm64']}",
              flush=True)
        print("   " + " | ".join(f"{n} {row[n + '_ms']:6.2f} ms ({flops / row[n + '_ms'] / 1e9:4.1f} TF/s)"
                                 for n, _ in variants), flush=True)
        print(f"   stages tm32: plan {row['stage_plan_ms']:.2f} gather {row['stage_gather_ms']:.2f} "
              f"gate_up {row['stage_gate_up_ms']:.2f} down {row['stage_down_ms']:.2f} ms", flush=True)

    for layer in picks:
        tune_layer(layer)


def small_weights(args, weights, mx, rows):
    """Cold per-call cost of GLM's small decode projections (m=1), all layers per pass."""
    import numpy as np

    from mlx2.runtime.gguf.weights import GGUFWeights  # noqa: F401  (type context)

    tensors = weights.tensors
    mla = [i for i in range(45) if f"blk.{i}.attn_kv_a_mqa.weight" in tensors]
    kda = [i for i in range(45) if f"blk.{i}.ssm_beta.weight" in tensors]
    kinds = {
        "hc_attn_fn F32 16384->24 (x45)": [f"blk.{i}.hc_attn_fn" for i in range(45)],
        "hc_ffn_fn F32 16384->24 (x45)": [f"blk.{i}.hc_ffn_fn" for i in range(45)],
        "router ffn_gate_inp F32 4096->144 (x42)": [f"blk.{i}.ffn_gate_inp.weight" for i in range(3, 45)],
        "indexer.proj (x11)": [f"blk.{i}.indexer.proj.weight" for i in mla],
        "indexer.attn_k (x11)": [f"blk.{i}.indexer.attn_k.weight" for i in mla],
        "indexer.attn_q_b (x11)": [f"blk.{i}.indexer.attn_q_b.weight" for i in mla],
        "kpool_gate F32 (x11)": [f"blk.{i}.indexer.kpool_gate" for i in mla],
        "mla attn_q_a (x11)": [f"blk.{i}.attn_q_a.weight" for i in mla],
        "mla attn_kv_a_mqa (x11)": [f"blk.{i}.attn_kv_a_mqa.weight" for i in mla],
        "kda ssm_beta (x34)": [f"blk.{i}.ssm_beta.weight" for i in kda],
    }
    for label, names in kinds.items():
        names = [n for n in names if n in tensors]
        if not names:
            print(f"small {label:42s} absent", flush=True)
            continue
        fns = [weights.linear(n) for n in names]
        xs = [mx.random.normal((1, tensors[n].numpy_shape[-1])).astype(mx.float32) for n in names]
        nbytes = sum(tensors[n].nbytes for n in names)
        mx.eval([f(x) for f, x in zip(fns, xs)])
        t = min(_timed_once(lambda fns=fns, xs=xs: [f(x) for f, x in zip(fns, xs)], mx) for _ in range(5))
        rows.append({"label": f"small {label}", "calls": len(names), "ms_per_pass": t * 1e3,
                     "us_per_call": t * 1e6 / len(names), "gb_s": nbytes / t / 1e9, "bytes": nbytes,
                     "ggml_type": int(tensors[names[0]].ggml_type)})
        print(f"small {label:42s} t{tensors[names[0]].ggml_type:<3d} {t*1e3:6.2f} ms/pass "
              f"{t*1e6/len(names):6.1f} us/call {nbytes/t/1e9:6.1f} GB/s", flush=True)
    # grouped MLA k_b / v_b through the row kernels
    for name in ("attn_k_b", "attn_v_b"):
        names = [f"blk.{i}.{name}.weight" for i in mla]
        x = mx.random.normal((1, 1, 64, 512 if name == "attn_k_b" else 256)).astype(mx.float32)
        # attn_k_b: [64 groups, 512 out, 256 in] consumes q_nope; attn_v_b: [64, 256, 512]
        shape = tensors[names[0]].numpy_shape
        x = mx.random.normal((1, 1, shape[0], shape[2])).astype(mx.float32)
        nbytes = sum(tensors[n].nbytes for n in names)
        fn = lambda names=names, x=x: [weights.grouped(x, n, groups=64) for n in names]
        mx.eval(fn())
        t = min(_timed_once(fn, mx) for _ in range(5))
        rows.append({"label": f"small grouped {name}", "calls": len(names), "ms_per_pass": t * 1e3,
                     "us_per_call": t * 1e6 / len(names), "gb_s": nbytes / t / 1e9})
        print(f"small grouped {name:34s} {t*1e3:6.2f} ms/pass {t*1e6/len(names):6.1f} us/call "
              f"{nbytes/t/1e9:6.1f} GB/s", flush=True)
    del np


def repack_q3n(raw, rows_, k):
    """Lossless Q3_K -> Q3N container (kblock_metal Q3N_FORMAT), numpy, CPU."""
    import numpy as np

    p = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 110)
    g = np.arange(16)
    lo = (p[:, 96 + g % 8] >> (4 * (g // 8))) & 15
    hi = (p[:, 104 + g % 4] >> (2 * (g // 4))) & 3
    sc = ((lo | (hi << 4)).astype(np.int16) - 32).astype(np.int8)            # [blocks, 16]
    i = np.arange(256)
    q = ((p[:, 32 + (i // 128) * 32 + i % 32] >> (2 * ((i // 32) % 4))) & 3).astype(np.int16)
    q -= 4 * (1 - ((p[:, i % 32] >> (i // 32)) & 1)).astype(np.int16)        # [blocks, 256] in -4..3
    code = (q + 4).astype(np.uint8).reshape(-1, 8, 32)
    nibbles = (code[:, :, :16] | (code[:, :, 16:] << 4)).reshape(-1, 128)
    out = np.zeros((p.shape[0], 148), np.uint8)
    out[:, :128] = nibbles
    out[:, 128:144] = sc.view(np.uint8)
    out[:, 144:146] = p[:, 108:110]
    return out.reshape(rows_, (k // 256) * 148)


def q3n_bench(args, weights, kblock_metal, mx, rows):
    """Cold decode: Q3_K v1 rows kernel vs the same tensors in the Q3N container."""
    from dataclasses import replace

    import numpy as np

    from mlx2.runtime.gguf.packed_metal import PackedMatrix

    linear = [i for i in range(45) if f"blk.{i}.ssm_beta.weight" in weights.tensors]
    kinds = {"attn_q Q3_K (KDA x34)": [f"blk.{i}.attn_q.weight" for i in linear],
             "ffn_up_shexp Q3_K (x42)": [f"blk.{i}.ffn_up_shexp.weight" for i in range(3, 45)]}
    for label, names in kinds.items():
        orig = [weights.packed(n) for n in names]
        packs = []
        for w in orig:
            n, k = w.shape
            data = repack_q3n(np.array(w.data).tobytes(), n, k)
            tensor = replace(w.tensor, ggml_type=kblock_metal.Q3N_FORMAT, nbytes=int(data.size))
            # Research container: production row_bytes() does not admit it, so build directly.
            pack = object.__new__(PackedMatrix)
            pack.tensor, pack.shape, pack.ggml_type = tensor, w.shape, tensor.ggml_type
            pack.row_bytes, pack.data = data.shape[1], mx.array(data.reshape(-1))
            packs.append(pack)
        mx.eval([p.data for p in packs])
        x = mx.random.normal((1, orig[0].shape[1])).astype(mx.float32)
        a = np.array(kblock_metal.kblock_dense(x, orig[0], vec=1))
        b = np.array(kblock_metal.kblock_dense(x, packs[0], vec=1))
        mism = int(np.count_nonzero(a.view(np.int32) != b.view(np.int32)))
        res = {}
        for name, ws in (("q3k", orig), ("q3n", packs), ("q3k", orig), ("q3n", packs)):
            fn = lambda ws=ws, x=x: [kblock_metal.kblock_dense(x, w, vec=1) for w in ws]
            mx.eval(fn())
            t = min(_timed_once(fn, mx) for _ in range(5))
            res[name] = min(res.get(name, 1e9), t)
        nb = {"q3k": sum(w.tensor.nbytes for w in orig), "q3n": sum(p.tensor.nbytes for p in packs)}
        elems = sum(w.shape[0] * w.shape[1] for w in orig)
        row = {"label": f"q3n {label}", "bitwise_mismatches_vs_q3k_v1": mism, "outputs": int(a.size)}
        for name in ("q3k", "q3n"):
            row[f"{name}_ms_per_call"] = res[name] * 1e3 / len(names)
            row[f"{name}_gb_s"] = nb[name] / res[name] / 1e9
            row[f"{name}_gelem_s"] = elems / res[name] / 1e9
        rows.append(row)
        print(f"q3n {label:26s} Q3_K {row['q3k_ms_per_call']:.4f} ms/call {row['q3k_gb_s']:6.1f} GB/s "
              f"{row['q3k_gelem_s']:6.1f} Gelem/s | Q3N {row['q3n_ms_per_call']:.4f} ms/call {row['q3n_gb_s']:6.1f} GB/s "
              f"{row['q3n_gelem_s']:6.1f} Gelem/s | mism vs Q3_K v1 {mism}/{a.size}", flush=True)


def classify(args, weights, kblock_metal, packed_moe, mx, rows):
    """Exactness class evidence per format (mlx-uag exactness-tolerance-classes)."""
    from dataclasses import replace

    import numpy as np

    from mlx2.runtime.gguf.k_blocks import decode_k_blocks
    from mlx2.runtime.gguf.packed_metal import PackedMatrix

    def fp64_matrix(w):
        raw = np.array(w.data)
        return decode_k_blocks(raw.tobytes(), w.ggml_type).astype(np.float64).reshape(w.shape)

    def ulp(a, b):
        ai, bi = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
        ai = np.where(ai < 0, -(ai & 0x7FFFFFFF), ai)
        bi = np.where(bi < 0, -(bi & 0x7FFFFFFF), bi)
        return np.abs(ai - bi)

    picks = ["blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.5.attn_v.weight",
             "blk.0.attn_output.weight", "blk.0.ssm_f_b.weight"]
    mats = [(n, weights.packed(n)) for n in picks]
    head = weights.packed("output.weight")  # Q6_K: classify its first 8192 rows
    sliced = PackedMatrix(head.data[: 8192 * head.row_bytes], replace(head.tensor, shape=(head.shape[1], 8192),
                                                                               nbytes=8192 * head.row_bytes))
    mats.append(("output.weight[:8192]", sliced))
    rng = np.random.default_rng(11)
    for name, w in mats:
        x = rng.standard_normal((1, w.shape[1])).astype(np.float32)
        y64 = fp64_matrix(w) @ x[0].astype(np.float64)
        ref = np.array(w(mx.array(x)))[0]
        ref_rms = float(np.sqrt(np.mean((ref.astype(np.float64) - y64) ** 2)))
        for v in args.variants.split(","):
            out = np.array(kblock_metal.kblock_dense(mx.array(x), w, **VARIANTS[v]))[0]
            rms = float(np.sqrt(np.mean((out.astype(np.float64) - y64) ** 2)))
            u = ulp(out, ref)
            v1_out = np.array(kblock_metal.kblock_dense(mx.array(x), w, **VARIANTS["v1"]))[0]
            row = {"label": f"classify {name} {v}", "ggml_type": w.ggml_type,
                   "bitwise_mismatches_vs_v1": int(np.count_nonzero(out.view(np.int32) != v1_out.view(np.int32))),
                   "bitwise_mismatches": int(np.count_nonzero(out.view(np.int32) != ref.view(np.int32))),
                   "outputs": int(out.size), "fp64_rms_ratio": rms / ref_rms if ref_rms else None,
                   "max_ulp_vs_reference": int(u.max()), "p99_ulp_vs_reference": float(np.percentile(u, 99))}
            rows.append(row)
            print(f"{row['label']:46s} t{w.ggml_type:<3d} mismatch {row['bitwise_mismatches']:6d}/{row['outputs']:<6d}"
                  f" fp64 ratio {row['fp64_rms_ratio']:.4f} maxULP {row['max_ulp_vs_reference']}"
                  f" vs v1 {row['bitwise_mismatches_vs_v1']}", flush=True)
    # Grouped launches against separate packed-reference projections.
    for names in (["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                   "blk.0.ssm_f_a.weight", "blk.0.ssm_g_a.weight", "blk.0.ssm_beta.weight"],
                  ["blk.0.ssm_f_b.weight", "blk.0.ssm_g_b.weight"]):
        ws = [weights.packed(n) for n in names]
        xs = [mx.array(rng.standard_normal((1, w.shape[1])).astype(np.float32)) for w in ws]
        if len({w.shape[1] for w in ws}) == 1:
            xs = [xs[0]] * len(ws)
        y = np.array(kblock_metal.kblock_multi(xs, ws))[0]
        start = 0
        for name, w, x in zip(names, ws, xs):
            ref = np.array(w(x))[0]
            got = y[start:start + w.shape[0]]
            single = np.array(kblock_metal.kblock_dense(x, w))[0]
            start += w.shape[0]
            row = {"label": f"classify multi {name}",
                   "mismatch_vs_reference": int(np.count_nonzero(got.view(np.int32) != ref.view(np.int32))),
                   "mismatch_vs_single_kblock": int(np.count_nonzero(got.view(np.int32) != single.view(np.int32))),
                   "outputs": int(got.size)}
            rows.append(row)
            print(f"{row['label']:46s} vs reference {row['mismatch_vs_reference']:6d}  vs single kblock "
                  f"{row['mismatch_vs_single_kblock']:6d} /{row['outputs']}", flush=True)
    # Routed experts, one layer, top-8 distinct experts.
    gate, up, down = (weights.packed(f"blk.3.ffn_{p}_exps.weight") for p in ("gate", "up", "down"))
    x = mx.array(rng.standard_normal((1, gate.shape[2])).astype(np.float32))
    ids = mx.random.permutation(gate.shape[0])[:8].reshape(1, 8).astype(mx.uint32)
    scores = mx.random.uniform(shape=(1, 8))
    ref = np.array(packed_moe(x, ids, scores, gate, up, down))[0]
    for v in args.variants.split(","):
        out = np.array(kblock_metal.kblock_moe(x, ids, scores, gate, up, down, **VARIANTS[v]))[0]
        row = {"label": f"classify routed_moe {v}",
               "bitwise_mismatches": int(np.count_nonzero(out.view(np.int32) != ref.view(np.int32))),
               "outputs": int(out.size), "max_ulp_vs_reference": int(ulp(out, ref).max())}
        rows.append(row)
        print(f"{row['label']:46s}      mismatch {row['bitwise_mismatches']:6d}/{row['outputs']:<6d} maxULP {row['max_ulp_vs_reference']}", flush=True)


def cold(args, weights, kblock_metal, packed_moe, mx, rows):
    """Per-call time over every layer's tensor of a kind (several hundred MB per pass)."""
    import numpy as np

    linear = [i for i in range(45) if f"blk.{i}.ssm_beta.weight" in weights.tensors]
    kinds = {
        "cold attn_q Q3_K (KDA)": [f"blk.{i}.attn_q.weight" for i in linear],
        "cold attn_output Q4_K (KDA)": [f"blk.{i}.attn_output.weight" for i in linear],
        "cold ffn_up_shexp Q3_K": [f"blk.{i}.ffn_up_shexp.weight" for i in range(3, 45)],
        "cold ffn_down_shexp Q4_K": [f"blk.{i}.ffn_down_shexp.weight" for i in range(3, 45)],
    }
    for label, names in kinds.items():
        ws = [weights.packed(n) for n in names]
        x = mx.random.normal((1, ws[0].shape[1])).astype(mx.float32)
        nbytes = sum(w.tensor.nbytes for w in ws)
        base = [np.array(kblock_metal.kblock_dense(x, w, vec=1)).view(np.uint32) for w in ws]
        for v in args.variants.split(","):
            fn = lambda v=v, x=x, ws=ws: [kblock_metal.kblock_dense(x, w, **VARIANTS[v]) for w in ws]
            outs = fn()
            mx.eval(outs)
            same = all(np.array_equal(np.array(o).view(np.uint32), b) for o, b in zip(outs, base))
            t = min(_timed_once(fn, mx) for _ in range(5))
            rows.append({"label": f"{label} {v}", "bytes": nbytes, "kblock_ms": t * 1e3 / len(ws),
                         "kblock_gb_s": nbytes / t / 1e9, "calls": len(ws), "bitwise_vs_v1": same})
            print(f"{label + ' ' + v:36s} {t*1e3/len(ws):7.3f} ms/call {nbytes/t/1e9:6.1f} GB/s", flush=True)
    layers = range(3, 45)
    banks = [[weights.packed(f"blk.{i}.ffn_{p}_exps.weight") for p in ("gate", "up", "down")] for i in layers]
    experts, _, k = banks[0][0].shape
    per_expert = sum(w.tensor.nbytes for w in banks[0]) / experts
    x = mx.random.normal((1, k)).astype(mx.float32)
    ids = [mx.random.permutation(experts)[:8].reshape(1, 8).astype(mx.uint32) for _ in layers]
    scores = mx.random.uniform(shape=(1, 8))
    base = [np.array(y).view(np.uint32) for y in
            [kblock_metal.kblock_moe(x, i, scores, *b, vec=1) for i, b in zip(ids, banks)]]
    for v in args.variants.split(","):
        fn = lambda v=v: [kblock_metal.kblock_moe(x, i, scores, *b, **VARIANTS[v]) for i, b in zip(ids, banks)]
        outs = fn()
        mx.eval(outs)
        same = all(np.array_equal(np.array(o).view(np.uint32), b) for o, b in zip(outs, base))
        t = min(_timed_once(fn, mx) for _ in range(5))
        nbytes = 8 * per_expert * len(banks)
        rows.append({"label": f"cold routed_moe {v}", "bytes": nbytes, "kblock_ms": t * 1e3 / len(banks),
                     "kblock_gb_s": nbytes / t / 1e9, "calls": len(banks), "bitwise_vs_v1": same})
        print(f"{'cold routed_moe ' + v:36s} {t*1e3/len(banks):7.3f} ms/call {nbytes/t/1e9:6.1f} GB/s", flush=True)


def _timed_once(fn, mx):
    t0 = time.perf_counter()
    mx.eval(fn())
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--kblock-rows", type=lambda s: [int(x) for x in s.split(",")], default=[4],
                        help="ROWS variants of the K-block kernels to time against the reference")
    parser.add_argument("--kblock-vec", type=lambda s: [int(x) for x in s.split(",")], default=[0],
                        help="VEC variants (0 byte loads, 1 block-stride-aligned word loads)")
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--cold", action="store_true",
                        help="cycle every layer's tensor of each kind per call so reads miss the system cache")
    parser.add_argument("--classify", action="store_true",
                        help="exactness classes: bitwise vs packed reference, fp64-RMS ratio, max ULP")
    parser.add_argument("--prefill", type=int, default=0,
                        help="rows for the NAX prefill-tile comparison (0 = skip)")
    parser.add_argument("--small", action="store_true", help="cold per-call cost of the small decode projections")
    parser.add_argument("--q3n", action="store_true", help="Q3_K vs the lossless Q3N 4-bit container, cold")
    parser.add_argument("--policy-row", default="kblock_rows", help="GGUFWeights row_policy for --small")
    parser.add_argument("--routing", type=Path, default=None,
                        help="routing.npz from the profiler's --capture-routing: MoE tile tuning")
    parser.add_argument("--moe-v2", action="store_true",
                        help="with --prefill: segmented expert prefill v2 against the current kernel")
    parser.add_argument("--reg-tmr", type=int, default=2, help="register tile: 16-row tiles per simdgroup")
    parser.add_argument("--lowrank", action="store_true", help="cold KDA ssm_f_b/ssm_g_b multi launch")
    parser.add_argument("--hc-decode", action="store_true", help="one token's HC work over 45 layers")
    parser.add_argument("--moe-union", action="store_true", help="multi-row MoE: per-slot vs expert union")
    parser.add_argument("--tok-share", action="store_true",
                        help="cold W-row decode projections: per-token slices vs token-shared chunk law")
    parser.add_argument("--tok-widths", type=lambda v: [int(t) for t in v.split(",")], default=[2, 4, 8])
    parser.add_argument("--prefill-scratch", action="store_true",
                        help="decode-once dense prefill (v2d) vs the v2 tile at --prefill rows")
    parser.add_argument("--prefill-v2", action="store_true",
                        help="with --prefill: compare tile v2 against the current tile")
    parser.add_argument("--variants", default="v1",
                        help="comma list of K-block variants: v0 v1 v2 exact (vec/exact switches)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx

    from mlx2.adapters.glm53_gguf_artifact import FILENAME, inspect_artifact
    from mlx2.runtime.gguf import kblock_metal
    from mlx2.runtime.gguf.packed_metal import PackedMatrix, grouped_linear, packed_moe
    from mlx2.runtime.gguf.weights import GGUFWeights

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])  # as the server does
    artifact = inspect_artifact(MODEL)
    weights = GGUFWeights(
        MODEL / FILENAME,
        row_policy=args.policy_row,
        expected_stat=artifact["weight_stat"],
        native_read_view=MODEL / "mlx2-native-packed.safetensors",
        source_sha256=artifact["identity"]["weight_sha256"],
    )
    key = mx.random.key(7)
    rows = []

    def record(label, ttype, nbytes, m, ref_fn, new_fn):
        ref, new = ref_fn(), new_fn()
        mx.eval(ref, new)
        t_ref, t_new = timed(ref_fn, args.reps, mx), timed(new_fn, args.reps, mx)
        row = {"label": label, "ggml_type": ttype, "rows": m, "bytes": nbytes,
               **error(ref, new, mx), "finite": bool(mx.all(mx.isfinite(new)).item()),
               "reference_ms": t_ref * 1e3, "kblock_ms": t_new * 1e3,
               "reference_gb_s": nbytes / t_ref / 1e9, "kblock_gb_s": nbytes / t_new / 1e9,
               "speedup": t_ref / t_new}
        rows.append(row)
        print(f"{label:34s} t{ttype:<3d} m={m:<2d} ref {t_ref*1e3:8.3f} ms {row['reference_gb_s']:6.1f} GB/s"
              f" | kblock {t_new*1e3:8.3f} ms {row['kblock_gb_s']:6.1f} GB/s  x{row['speedup']:5.2f}"
              f"  rel {row['rel_l2']:.2e}", flush=True)

    for name in DENSE:
        w = weights.packed(name)
        k = w.shape[1]
        for m in args.rows:
            x = mx.random.normal((m, k), key=mx.random.split(key)[0]).astype(mx.float32)
            for r in args.kblock_rows:
                for v in args.kblock_vec:
                    record(f"{name} R{r}V{v}", w.ggml_type, w.tensor.nbytes, m, lambda w=w, x=x: w(x),
                           lambda w=w, x=x, r=r, v=v: kblock_metal.kblock_dense(x, w, rows=r, vec=v))
    from dataclasses import replace

    for name, groups in GROUPED:
        base = weights.packed(name)
        t = base.tensor
        shape = t.numpy_shape if len(t.numpy_shape) == 3 else (groups, t.numpy_shape[0] // groups, t.numpy_shape[1])
        w = PackedMatrix(base.data, replace(t, shape=shape[::-1]))
        g, _, k = w.shape
        for m in args.rows:
            x = mx.random.normal((m, g, k)).astype(mx.float32)
            record(name, w.ggml_type, t.nbytes, m, lambda w=w, x=x: grouped_linear(x, w),
                   lambda w=w, x=x: kblock_metal.kblock_grouped(x, w))
    gate, up, down = (weights.packed(f"{MOE}.ffn_{p}_exps.weight") for p in ("gate", "up", "down"))
    experts, _, k = gate.shape
    per_expert = (gate.tensor.nbytes + up.tensor.nbytes + down.tensor.nbytes) / experts
    for m in args.rows:
        x = mx.random.normal((m, k)).astype(mx.float32)
        ids = mx.stack([mx.random.permutation(experts)[:8] for _ in range(m)]).astype(mx.uint32)
        scores = mx.random.uniform(shape=(m, 8))
        distinct = len({int(i) for i in ids.reshape(-1).tolist()})
        for r in args.kblock_rows:
            for v in args.kblock_vec:
                record(f"{MOE}.routed_moe(top8) R{r}V{v}", gate.ggml_type, distinct * per_expert, m,
                       lambda x=x, ids=ids, scores=scores: packed_moe(x, ids, scores, gate, up, down),
                       lambda x=x, ids=ids, scores=scores, r=r, v=v: kblock_metal.kblock_moe(
                           x, ids, scores, gate, up, down, rows=r, vec=v))
    # Shared-expert fold against the unfused routed + shared path.
    gs, us, ds_ = (weights.packed(f"{MOE}.ffn_{p}_shexp.weight") for p in ("gate", "up", "down"))
    shared_bytes = gs.tensor.nbytes + us.tensor.nbytes + ds_.tensor.nbytes

    def unfused(x, ids, scores):
        routed = kblock_metal.kblock_moe(x, ids, scores, gate, up, down)
        g = mx.minimum(kblock_metal.kblock_dense(x, gs), 10)
        u = mx.clip(kblock_metal.kblock_dense(x, us), -10, 10)
        return routed + kblock_metal.kblock_dense((g * mx.sigmoid(g)) * u, ds_)

    for m in args.rows:
        x = mx.random.normal((m, k)).astype(mx.float32)
        ids = mx.stack([mx.random.permutation(experts)[:8] for _ in range(m)]).astype(mx.uint32)
        scores = mx.random.uniform(shape=(m, 8))
        distinct = len({int(i) for i in ids.reshape(-1).tolist()})
        record(f"{MOE}.routed+shared fold", gate.ggml_type, distinct * per_expert + shared_bytes, m,
               lambda x=x, ids=ids, scores=scores: unfused(x, ids, scores),
               lambda x=x, ids=ids, scores=scores: kblock_metal.kblock_moe_shared(
                   x, ids, scores, gate, up, down, gs, us, ds_))
    if args.moe_union:
        moe_union(args, weights, kblock_metal, mx, rows)
    if args.hc_decode:
        hc_decode(args, weights, mx, rows)
    if args.lowrank:
        lowrank(args, weights, kblock_metal, mx, rows)
    if args.tok_share:
        tok_share(args, weights, kblock_metal, mx, rows)
    if args.q3n:
        q3n_bench(args, weights, kblock_metal, mx, rows)
    if args.small:
        small_weights(args, weights, mx, rows)
    if args.routing:
        moe_tune(args, weights, mx, rows)
    if args.prefill and args.moe_v2:
        moe_v2(args, weights, mx, rows)
    if args.prefill and args.prefill_scratch:
        prefill_scratch(args, weights, mx, rows)
    elif args.prefill and args.prefill_v2:
        prefill_v2(args, weights, mx, rows)
    elif args.prefill:
        prefill_tiles(args, weights, mx, rows)
    if args.classify:
        classify(args, weights, kblock_metal, packed_moe, mx, rows)
    if args.cold:
        cold(args, weights, kblock_metal, packed_moe, mx, rows)
    result = {"schema": "mlx2.glm53-kblock-microbench.v1", "source_revision": args.source_revision,
              "device": mx.device_info(), "reps": args.reps, "rows": rows}
    (args.out / "microbench.json").write_text(json.dumps(result, indent=1) + "\n")
    weights.close()


if __name__ == "__main__":
    main()
