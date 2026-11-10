#!/usr/bin/env python3
"""Fit DS4.1 Source ridge lookahead heads from a routing capture (K3.1 residual-ridge method).

For token t and target layer L, depth d: hidden = x_{L-d}(t) (router input at layer L-d),
base = router_L(hidden) (the untrained lookahead), target = router_L(x_L(t)) (exact full scores).
Head: target ~ base + hidden @ W.T + b (fit_ridge_head). Train on the train split, report held-out
recall@6 of the true top-6 (selection = top-6 of score + correction bias) for untrained vs trained.
Writes heads via save_heads (checkpoint-bound) plus heads-eval.json.

usage: train_heads.py CAPTURE_DIR LITE_DIR OUT_HEADS.npz [--depths 1,2] [--lam 100]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_sim import read_capture  # noqa: E402
from mlx2.adapters.deepseek_v41_source_stream import fit_ridge_head, save_heads  # noqa: E402

LAYERS, E, K, H = 40, 384, 6, 5120
BINDING = "c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb"


def softplus(z):
    return np.logaddexp(0.0, z)


def router(gate, x):
    return np.sqrt(softplus(x @ gate.T))


def topk(s, k=K):
    return np.argpartition(-s, k - 1, axis=1)[:, :k]


def recall(pred, true):
    return float(np.mean([len(set(p) & set(t)) / K for p, t in zip(pred, true)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir"); ap.add_argument("lite"); ap.add_argument("out")
    ap.add_argument("--depths", default="1,2"); ap.add_argument("--lam", type=float, default=100.0)
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]
    replies = [json.loads(line) for line in open(f"{a.capture_dir}/replies.jsonl")]
    windows = [(r["t0_ns"], r["t1_ns"], r["split"]) for r in replies]
    # Pass 1: per-layer x memmaps (fp16) aligned by token index, plus split labels.
    tmp = f"{a.capture_dir}/x_by_layer"
    os.makedirs(tmp, exist_ok=True)
    n_tok = sum(rows for layer, rows, *_ in read_capture(f"{a.capture_dir}/capture.bin") if layer == 0)
    xs = [np.lib.format.open_memmap(f"{tmp}/L{L:02d}.npy", "w+", np.float16, (n_tok, H)) for L in range(LAYERS)]
    split = np.zeros(n_tok, np.int8)       # 0 train, 1 heldout, -1 outside any request window
    pos = [0] * LAYERS
    for layer, rows, t, _ids, _s, x in read_capture(f"{a.capture_dir}/capture.bin", keep_x=True):
        p = pos[layer]
        xs[layer][p:p + rows] = x
        if layer == 0:
            w = next((w for w in windows if w[0] <= t <= w[1]), None)
            split[p:p + rows] = -1 if w is None else (1 if w[2] == "heldout" else 0)
        pos[layer] = p + rows
    if len(set(pos)) != 1:
        raise SystemExit(f"layer row counts differ: {sorted(set(pos))}")
    for m in xs:
        m.flush()
    tr, ho = np.where(split == 0)[0], np.where(split == 1)[0]
    heads, report = {}, {"tokens": n_tok, "train": len(tr), "heldout": len(ho), "lam": a.lam, "rows": []}
    import mlx.core as mx
    for L in range(1, LAYERS):
        lite = mx.load(f"{a.lite}/layer_{L:02d}.safetensors")
        gate = np.array(lite["gate.weight"].astype(mx.float32)).astype(np.float64)
        bias = np.array(lite["gate.bias"]).astype(np.float64)
        xl = np.asarray(xs[L], np.float64)
        target = router(gate, xl)
        true = topk(target + bias)
        for d in depths:
            if L - d < 0:
                continue
            hid = np.asarray(xs[L - d], np.float64)
            base = router(gate, hid)
            w, b = fit_ridge_head(hid[tr], base[tr], target[tr], lam=a.lam)
            pred_u = topk(base[ho] + bias); pred_t = topk(base[ho] + hid[ho] @ w.T.astype(np.float64) + b + bias)
            row = {"layer": L, "depth": d, "recall_untrained": round(recall(pred_u, true[ho]), 4),
                   "recall_trained": round(recall(pred_t, true[ho]), 4)}
            report["rows"].append(row); heads[(L, d)] = (w, b)
            print(row, flush=True)
    for d in depths:
        rs = [r for r in report["rows"] if r["depth"] == d]
        report[f"mean_d{d}"] = {"untrained": round(float(np.mean([r["recall_untrained"] for r in rs])), 4),
                                "trained": round(float(np.mean([r["recall_trained"] for r in rs])), 4)}
    report["heads_sha256"] = save_heads(a.out, heads, binding=BINDING, hidden_size=H, experts=E, lam=a.lam)
    json.dump(report, open(a.out + ".eval.json", "w"), indent=1)
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
