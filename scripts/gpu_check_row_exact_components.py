#!/usr/bin/env python3
"""Metal bit-exactness of Flash-Next verify-window components vs one-row decode.

Synthetic weights at the served Flash-Next shapes.  For each component the
verify form (R rows at once) is compared bit for bit with R one-row serial
calls, for the stock verify path and, where one exists, the row-exact path:

* ``qmv``   every served projection shape: stock ``QuantizedLinear`` at M=R vs
            the row-exact kernel (``row_exact_qmv``) vs R stock M=1 calls.
* ``gdn``   the Qwen4 GDN core: R sequential one-token fused decode steps vs
            the fused verify kernel (snapshot mode) and the compact replay
            verify + reconstruct (every accepted prefix).
* ``sdpa``  attention rows: R-query causal SDPA vs R one-query SDPAs over each
            row's own key prefix, at several cached lengths.
* ``norm``  row-invariance of the reductions the window runs batched
            (fast RMSNorm, grouped fp32 RMSNorm, HC stream mean, logsumexp).

  scratchpad/gpuq.sh l5-comp PYTHONPATH=src MLX_ENABLE_TF32=0 .venv/bin/python \\
      scripts/gpu_check_row_exact_components.py --i-own-the-gpu --out comp.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROWS = (2, 3, 4, 8, 9, 12, 16, 17)

# (name, K, N, bits, group_size) at the served Flash-Next geometry.
SHAPES = (
    ("attn.q_proj", 2560, 12288, 4, 64),
    ("attn.k_proj", 2560, 512, 4, 64),
    ("attn.o_proj", 6144, 2560, 4, 64),
    ("attn.index_qk", 2560, 640, 4, 64),
    ("gdn.in_proj_qkv", 2560, 10240, 4, 64),
    ("gdn.in_proj_z", 2560, 6144, 4, 64),
    ("gdn.in_proj_ab", 2560, 48, 4, 64),
    ("gdn.in_fused", 2560, 16480, 4, 64),
    ("gdn.out_proj", 6144, 2560, 4, 64),
    ("hc.mix_down", 10240, 320, 4, 64),
    ("hc.mix_up", 320, 10240, 4, 64),
    ("hc.inject", 10240, 4, 4, 64),
    ("moe.router_gate", 2560, 512, 8, 64),
    ("moe.shared_gate_up", 2560, 640, 4, 64),
    ("moe.shared_down", 640, 2560, 4, 64),
    ("moe.shared_expert_gate", 2560, 1, 8, 64),
    ("lm_head", 2560, 248320, 4, 64),
)


def bits_equal(a, b):
    import mlx.core as mx

    if a.shape != b.shape:
        return False, -1
    if a.dtype == mx.bfloat16:
        a, b = a.view(mx.uint16), b.view(mx.uint16)
    elif a.dtype == mx.float32:
        a, b = a.view(mx.uint32), b.view(mx.uint32)
    diff = int(mx.sum(a != b).item())
    return diff == 0, diff


def check_qmv(rows_list, seed):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx2.runtime.models import row_exact_qmv as REQ

    out = []
    for name, k, n, bits, gs in SHAPES:
        mx.random.seed(seed)
        linear = nn.Linear(k, n, bias=False)
        linear.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        q = nn.QuantizedLinear.from_linear(linear, group_size=gs, bits=bits)
        q.set_dtype(mx.bfloat16)
        mx.eval(q.parameters())
        for rows in rows_list:
            x = (mx.random.normal((1, rows, k)) * 1.0).astype(mx.bfloat16)
            serial = mx.concatenate([q(x[:, r : r + 1]) for r in range(rows)], axis=1)
            stock = q(x)
            exact, route = REQ.quantized_linear(q, x)
            mx.eval(serial, stock, exact)
            s_ok, s_diff = bits_equal(stock, serial)
            e_ok, e_diff = bits_equal(exact, serial)
            out.append({"shape": name, "K": k, "N": n, "bits": bits, "rows": rows,
                        "fast": REQ.qmv_fast_layout(k, n, bits),
                        "stock_equal": s_ok, "stock_diff_elems": s_diff,
                        "row_exact_equal": e_ok, "row_exact_diff_elems": e_diff,
                        "route": route})
        del q, linear
        mx.clear_cache()
    return out


def check_gdn(rows_list, seed):
    import mlx.core as mx
    from mlx2.runtime.models import qwen4_fused_gdn as FG
    from mlx2.runtime.models import qwen4_fused_gdn_verify as FV

    FV.set_verify_max_steps(17)
    ty = FG.probe_qwen4_fused_gdn_decode(mx.bfloat16)
    out = []
    mx.random.seed(seed)
    conv_weight = (mx.random.normal((FG.CONV_DIM, FG.CONV_KERNEL, 1)) * 0.3).astype(mx.bfloat16)
    A_log = mx.random.uniform(-2, 1, (FG.NUM_VALUE_HEADS,)).astype(mx.float32)
    dt_bias = (mx.random.normal((FG.NUM_VALUE_HEADS,)) * 0.5).astype(mx.bfloat16)
    norm_weight = (1 + mx.random.normal((FG.VALUE_HEAD_DIM,)) * 0.1).astype(mx.bfloat16)
    for rows in rows_list:
        qkv = mx.random.normal((1, rows, FG.CONV_DIM)).astype(mx.bfloat16)
        z = mx.random.normal((1, rows, FG.VALUE_DIM)).astype(mx.bfloat16)
        b = mx.random.normal((1, rows, FG.NUM_VALUE_HEADS)).astype(mx.bfloat16)
        a = mx.random.normal((1, rows, FG.NUM_VALUE_HEADS)).astype(mx.bfloat16)
        conv0 = mx.random.normal((1, FG.CONV_KERNEL - 1, FG.CONV_DIM)).astype(mx.bfloat16)
        state0 = (mx.random.normal((1, FG.NUM_VALUE_HEADS, FG.VALUE_HEAD_DIM, FG.KEY_HEAD_DIM)) * 0.1).astype(mx.float32)
        mx.eval(qkv, z, b, a, conv0, state0)
        conv, state = conv0, state0
        outs, states, convs = [], [], []
        for r in range(rows):
            o, conv, state = FG.qwen4_fused_gdn_decode(
                qkv[:, r : r + 1], z[:, r : r + 1], b[:, r : r + 1], a[:, r : r + 1],
                conv, conv_weight, A_log, dt_bias, state, norm_weight, 1e-6, threadgroup_y=ty)
            mx.eval(o, conv, state)
            outs.append(o)
            states.append(state)
            convs.append(conv)
        serial_out = mx.concatenate(outs, axis=1)
        entry = {"rows": rows}
        vty = FV.probe_qwen4_fused_gdn_verify(mx.bfloat16, rows)
        if vty is not None:
            vo, vconv, vstate, snaps, csnaps = FV.qwen4_fused_gdn_verify(
                qkv, z, b, a, conv0, conv_weight, A_log, dt_bias, state0, norm_weight,
                1e-6, threadgroup_y=vty)
            mx.eval(vo, vconv, vstate, snaps, csnaps)
            entry["verify_out_equal"] = bits_equal(vo, serial_out)[0]
            entry["verify_final_state_equal"] = bits_equal(vstate, states[-1])[0]
            entry["verify_conv_equal"] = bits_equal(vconv, convs[-1])[0]
            entry["verify_snapshots_equal"] = all(
                bits_equal(snaps[:, p], states[p])[0] for p in range(rows - 1))
        rty = FV.probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, rows)
        if rty is not None:
            ro, rconv, rstate, keys, corr, decay = FV.qwen4_fused_gdn_replay_verify(
                qkv, z, b, a, conv0, conv_weight, A_log, dt_bias, state0, norm_weight,
                1e-6, threadgroup_y=rty)
            mx.eval(ro, rstate, keys, corr, decay)
            entry["replay_out_equal"] = bits_equal(ro, serial_out)[0]
            entry["replay_final_state_equal"] = bits_equal(rstate, states[-1])[0]
            recon = []
            for m in range(1, rows):
                rs = FV.qwen4_fused_gdn_reconstruct(state0, keys, corr, decay, m, threadgroup_y=rty)
                mx.eval(rs)
                recon.append(bits_equal(rs, states[m - 1])[0])
            entry["replay_reconstruct_equal"] = all(recon)
            entry["replay_reconstruct_equal_by_m"] = recon
        out.append(entry)
        mx.clear_cache()
    return out


def check_sdpa(rows_list, seed, contexts=(37, 1000, 1022, 1030, 4090, 5000)):
    import mlx.core as mx

    out = []
    mx.random.seed(seed)
    hq, hk, d = 24, 2, 256
    scale = d ** -0.5
    for ctx in contexts:
        for rows in rows_list:
            total = ctx + rows
            q = mx.random.normal((1, hq, rows, d)).astype(mx.bfloat16)
            k = mx.random.normal((1, hk, total, d)).astype(mx.bfloat16)
            v = mx.random.normal((1, hk, total, d)).astype(mx.bfloat16)
            causal = mx.arange(total)[None, :] <= (ctx + mx.arange(rows))[:, None]
            batched = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=causal[None, None])
            serial = mx.concatenate([
                mx.fast.scaled_dot_product_attention(
                    q[:, :, r : r + 1], k[:, :, : ctx + r + 1], v[:, :, : ctx + r + 1], scale=scale)
                for r in range(rows)], axis=2)
            mx.eval(batched, serial)
            ok, diff = bits_equal(batched, serial)
            out.append({"context": ctx, "rows": rows, "batched_equal": ok, "diff_elems": diff})
    return out


def check_norm(rows_list, seed):
    import mlx.core as mx
    import mlx.nn as nn

    out = []
    mx.random.seed(seed)
    w = (1 + mx.random.normal((2560,)) * 0.1).astype(mx.bfloat16)
    for rows in rows_list:
        x = mx.random.normal((1, rows, 10240)).astype(mx.bfloat16)
        checks = {}
        # fast RMSNorm (bf16, row width 2560 and 256)
        f = lambda t: mx.fast.rms_norm(t, w, 1e-6)
        xs = x[..., :2560]
        checks["rms_norm_bf16_2560"] = (f(xs), mx.concatenate([f(xs[:, r : r + 1]) for r in range(rows)], axis=1))
        # grouped fp32 RMSNorm (GroupRMSNorm fast arm: 4 streams of 2560)
        g = lambda t: mx.fast.rms_norm(t.astype(mx.float32).reshape(*t.shape[:-1], 4, 2560), None, 1e-6)
        checks["group_rms_fp32"] = (g(x), mx.concatenate([g(x[:, r : r + 1]) for r in range(rows)], axis=1))
        # GroupRMSNorm eager arm (used above the width gate)
        def eager(t):
            tf = t.astype(mx.float32).reshape(*t.shape[:-1], 4, 2560)
            return tf * mx.rsqrt(mx.mean(tf * tf, axis=-1, keepdims=True) + 1e-6)
        checks["group_rms_eager_vs_fast_serial"] = (
            mx.concatenate([eager(x[:, r : r + 1]) for r in range(rows)], axis=1),
            mx.concatenate([g(x[:, r : r + 1]) for r in range(rows)], axis=1))
        # HC stream mix: mean over the 4 streams
        s = x.reshape(1, rows, 4, 2560)
        wts = mx.sigmoid(mx.random.normal((1, rows, 4, 2560))).astype(mx.bfloat16)
        m = lambda a_, b_: mx.mean(a_ * b_, axis=-2)
        checks["hc_stream_mean"] = (m(wts, s), mx.concatenate([m(wts[:, r : r + 1], s[:, r : r + 1]) for r in range(rows)], axis=1))
        # logsumexp over the vocabulary (fp32)
        logits = mx.random.normal((rows, 248320)).astype(mx.float32)
        lse = lambda t: mx.logsumexp(t, axis=-1, keepdims=True)
        checks["logsumexp_vocab"] = (lse(logits), mx.concatenate([lse(logits[r : r + 1]) for r in range(rows)], axis=0))
        # PLE short conv: depthwise, kernel 4, dilation 3 over a 9-row state
        conv = nn.Conv1d(10240, 10240, 4, dilation=3, groups=10240, bias=False)
        conv.weight = (mx.random.normal(conv.weight.shape) * 0.2).astype(mx.bfloat16)
        state = mx.random.normal((1, 9, 10240)).astype(mx.bfloat16)
        seq = mx.concatenate([state, x], axis=1)
        batched = nn.silu(conv(seq))[:, -rows:, :]
        serial = mx.concatenate(
            [nn.silu(conv(seq[:, r : r + 10]))[:, -1:, :] for r in range(rows)], axis=1)
        checks["ple_depthwise_conv"] = (batched, serial)
        row = {"rows": rows}
        for key, (a_, b_) in checks.items():
            mx.eval(a_, b_)
            row[key] = bits_equal(a_, b_)[0]
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="qmv,gdn,sdpa,norm")
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qmv-seeds", default="0,1,2", help="seeds for the qmv section")
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("Metal run: pass --i-own-the-gpu under the GPU lock")
    import mlx.core as mx

    mx.set_cache_limit(4 << 30)
    rows_list = [int(v) for v in args.rows.split(",")]
    report = {
        "schema": "mlx2.gpu-check-row-exact-components.v1",
        "mlx": mx.__version__,
        "tf32": os.environ.get("MLX_ENABLE_TF32"),
        "device": {k: v for k, v in mx.metal.device_info().items() if k in ("architecture", "device_name")},
    }
    sections = {"qmv": check_qmv, "gdn": check_gdn, "sdpa": check_sdpa, "norm": check_norm}
    for name in args.only.split(","):
        if name == "qmv":
            report[name] = [
                dict(cell, seed=seed)
                for seed in (int(v) for v in args.qmv_seeds.split(","))
                for cell in check_qmv(rows_list, seed)
            ]
        else:
            report[name] = sections[name](rows_list, args.seed)
        Path(args.out).write_text(json.dumps(report, indent=1))
    summary = {}
    if "qmv" in report:
        cells = report["qmv"]
        summary["qmv"] = {
            "cells": len(cells),
            "row_exact_equal": sum(c["row_exact_equal"] for c in cells),
            "stock_equal": sum(c["stock_equal"] for c in cells),
            "row_exact_failures": [c for c in cells if not c["row_exact_equal"]][:10],
        }
    if "gdn" in report:
        summary["gdn"] = report["gdn"]
    if "sdpa" in report:
        cells = report["sdpa"]
        summary["sdpa"] = {"cells": len(cells), "batched_equal": sum(c["batched_equal"] for c in cells)}
    if "norm" in report:
        summary["norm"] = report["norm"]
    report["summary"] = summary
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
