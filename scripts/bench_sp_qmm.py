#!/usr/bin/env python3
"""Cache-cold small-M quantized matmul profile on the Qwen3.8-27B-oQ4e-mtp shapes.

GPU only (``--i-own-the-gpu``). For each projection shape and bit width in the
artifact, and each M in 1..32, times ``mx.quantized_matmul`` (bf16 activations,
affine, group size 64) with the weights streamed from DRAM: every timed op
rotates through about 1 GiB of distinct weight copies. Warm-cache numbers
overstate small-M headroom (qualification/runs/qmm-verify-flatten-tile8-20260919).

Each row reports microseconds per op, the ratio to M = 1, the effective weight
bandwidth (packed weights + bf16 scales and biases) and the MLX kernel path
that the fork's dispatch picks (mirrored from mlx/backend/metal/quantized.cpp
at 39400a0d4; M5 Max is applegpu_g17s).

With ``--arms stock,sp`` it also times the mlx2 small-M kernel
(``mlx2.runtime.models.sp_qmm``) interleaved with stock per repetition and
checks its error against stock and an fp32 reference.

  python scripts/bench_sp_qmm.py bw --i-own-the-gpu
  python scripts/bench_sp_qmm.py qmm --i-own-the-gpu --out profile.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# (N, K, bits, count per forward) from the safetensors headers.
SHAPES = {
    "gate_up_b4": (17408, 5120, 4, 128),
    "down_b4": (5120, 17408, 4, 37),
    "down_b5": (5120, 17408, 5, 27),
    "gdn_qkv_b4": (10240, 5120, 4, 48),
    "gdn_out_b5": (5120, 6144, 5, 48),
    "gdn_z_b4": (6144, 5120, 4, 27),
    "gdn_z_b5": (6144, 5120, 5, 21),
    "attn_q_b4": (12288, 5120, 4, 16),
    "attn_o_b4": (5120, 6144, 4, 9),
    "attn_o_b5": (5120, 6144, 5, 7),
    "attn_kv_b4": (1024, 5120, 4, 30),
    "gdn_ab_b5": (48, 5120, 5, 61),
    "lm_head_b4": (248320, 5120, 4, 1),
}
MS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
TARGET_BYTES = 1 << 30
GS = 64


def mlx_path(M, N, K, bits):
    """The kernel MLX 39400a0d4 dispatches on applegpu_g17s (transposed, 2-D w)."""
    if (bits in (4, 8) and M <= 16 and K % 128 == 0
            and (M >= 12 or (M >= 8 and N * K >= (1 << 24)))):
        return "qmv_nax"
    if K <= 2048 and N <= 2048:
        limit = 18
    elif K <= 4096 and N <= 4096:
        limit = 12
    else:
        limit = 16
    if M >= limit:
        return "qmm_splitk"
    if M == 1:
        return "qmv"
    tiles = -(-M // 8)
    return f"qmv_wide_x{tiles}"


def weight_bytes(N, K, bits):
    return N * K * bits // 8 + 2 * 2 * N * (K // GS)


def peak_bw(mx, reps=10):
    n = 1 << 28  # 1 GiB of float32
    a = mx.random.uniform(shape=(n,))
    mx.eval(a)
    out = {}
    for name, fn, traffic in (
        ("read_sum", lambda: mx.sum(a), 4 * n),
        ("copy", lambda: a * 2.0, 8 * n),
    ):
        for _ in range(2):
            mx.eval(fn())
        s = []
        for _ in range(reps):
            t0 = time.perf_counter(); mx.eval(fn()); s.append(time.perf_counter() - t0)
        out[name] = round(traffic / min(s) / 1e9, 1)
    a = None  # release the buffer (closures above reference it)
    mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bw", "qmm"])
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--arms", default="stock")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--serial", action="store_true",
                    help="chain the ops so they cannot overlap on the GPU")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")
    from sp_qmm_guard import SwapGuard
    import mlx.core as mx

    guard = SwapGuard()
    mx.set_cache_limit(4 << 30)
    sink = open(a.out, "a") if a.out else None

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n"); sink.flush()

    guard.timed()
    bw = peak_bw(mx)
    emit({"kind": "bandwidth", "GBps": bw, "mlx": mx.__version__,
          "device": mx.device_info()["architecture"]})
    if a.what == "bw":
        return
    arms = a.arms.split(",")
    sp_arms = [arm for arm in arms if arm.startswith("sp")]
    if sp_arms:
        from mlx2.runtime.models import sp_qmm

    def sp_cfg(arm):
        # "sp" = module defaults; "sp-NF-KS[-PF]" pins the tile (and prefetch).
        parts = arm.split("-")
        vals = [int(p) for p in parts[1:]] + [None] * 5
        return vals[0], vals[1], vals[2], vals[3] or 0, vals[4]
    ms = [int(m) for m in a.ms.split(",")]
    for name in a.shapes.split(","):
        N, K, bits, count = SHAPES[name]
        per = weight_bytes(N, K, bits)
        copies = max(2, TARGET_BYTES // per)
        guard.phase = "load"; guard.base = __import__("sp_qmm_guard").swapouts()
        guard.limit = guard.load_limit
        mats = []
        for c in range(copies):
            w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
            mats.append(mx.quantize(w, group_size=GS, bits=bits))
            mx.eval(mats[-1])
            del w
        mx.clear_cache()
        guard.timed()
        base = {}
        for m in ms:
            x = (mx.random.normal((m, K))).astype(mx.bfloat16)
            mx.eval(x)
            raw = {
                "stock": lambda xx, q: mx.quantized_matmul(
                    xx, *q, transpose=True, group_size=GS, bits=bits),
            }
            for arm in sp_arms:
                nf, ks, pf, diag, xl = sp_cfg(arm)
                raw[arm] = (lambda nf, ks, pf, diag, xl: lambda xx, q: sp_qmm.qmm(
                    xx, *q, group_size=GS, bits=bits, nf=nf, ks=ks, pf=pf, diag=diag,
                    xl=xl))(nf, ks, pf, diag, xl)
            fns = {arm: (lambda f: lambda q: f(x, q))(f) for arm, f in raw.items()}

            def run_all(arm):
                if not a.serial:
                    return [fns[arm](q) for q in mats]
                # Serial: each op's input depends on the previous output, so
                # kernels cannot overlap (as in a real forward).
                xx = x
                for q in mats:
                    y = raw[arm](xx, q)
                    xx = x + (y[:, :1] * 0).astype(x.dtype)
                return [xx]
            live = [arm for arm in fns if arm == "stock" or sp_qmm.supports(m, N, K, bits, GS)]
            errs = {}
            if len(live) > 1:
                q = mats[0]
                w32 = mx.dequantize(q[0], q[1].astype(mx.float32), q[2].astype(mx.float32),
                                    group_size=GS, bits=bits)
                ref = x.astype(mx.float32) @ w32.T
                scale = mx.max(mx.abs(ref)).item()
                st = fns["stock"](q).astype(mx.float32)
                for arm in live:
                    y = fns[arm](q).astype(mx.float32)
                    errs[arm] = {"vs_ref_max_abs": mx.max(mx.abs(y - ref)).item(),
                                 "vs_ref_max_rel": mx.max(mx.abs(y - ref)).item() / scale,
                                 "vs_stock_max_abs": mx.max(mx.abs(y - st)).item(),
                                 "ref_max_abs": scale}
                    if arm != "stock":
                        y2 = fns[arm](q)
                        errs[arm]["deterministic"] = bool(mx.array_equal(y2, fns[arm](q)).item())
                del ref, st, w32
            times = {arm: [] for arm in fns}
            for arm in live:
                for _ in range(2):
                    mx.eval(run_all(arm))
            for r in range(a.reps):
                order = live if r % 2 == 0 else live[::-1]
                for arm in order:
                    t0 = time.perf_counter()
                    mx.eval(run_all(arm))
                    times[arm].append((time.perf_counter() - t0) / copies)
            for arm in live:
                t = statistics.median(times[arm])
                base.setdefault(arm, t)
                emit({"kind": "qmm", "shape": name, "N": N, "K": K, "bits": bits,
                      "count": count, "m": m, "arm": arm,
                      "path": mlx_path(m, N, K, bits) if arm == "stock" else arm,
                      "us": round(t * 1e6, 1), "us_min": round(min(times[arm]) * 1e6, 1),
                      "ratio_m1": round(t / base[arm], 3),
                      "GBps": round(per / t / 1e9, 1), "copies": copies, "serial": a.serial,
                      **errs.get(arm, {})})
            x = None
        mats = None
        mx.clear_cache()
    emit({"kind": "swap", **guard.report()})


if __name__ == "__main__":
    main()
