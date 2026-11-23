"""Train a hybrid core (H40) on the long-context family and record held accuracy and cost.

  python -m research.ens.train_hybrid --pattern GGGGA --L 512 --k 1 --steps 4000 --out <dir>

Rows: research/ens/longctx.py (train seed 0, held seed 1, disjoint sequences). Loss: cross-entropy over the
64 value tokens at the ANS position. The receipt records the pattern, config, steps per second, peak memory,
held accuracy per hop count, and the checkpoint name (with a weight fingerprint), plus source and host, as
for core receipts. This is the fixed-hybrid baseline; the evolving arm builds on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import longctx as LC
from research.ens import vocab as V
from research.ens.hybrid import HybridConfig, HybridCore


def batch(rows):
    toks = mx.array([r[0] for r in rows])
    ans_idx = mx.array([r[2] for r in rows])
    ans = mx.array([r[3] for r in rows])
    return toks, ans_idx, ans


def value_logits(logits, ans_idx):
    B = logits.shape[0]
    return logits[mx.arange(B), ans_idx][:, V.V0:V.V0 + V.N_VALUES]


def evaluate(core, rows, bs=32):
    hit = 0
    for i in range(0, len(rows), bs):
        t, a, y = batch(rows[i:i + bs])
        hit += int(mx.sum(mx.argmax(value_logits(core(t), a), -1) == y).item())
    return hit / len(rows)


def train(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    cfg = HybridConfig(d=args.d, heads=args.heads, ffn=4 * args.d, seq_len=args.L, pattern=args.pattern,
                       sparse_block=args.sparse_block, sparse_topk=args.sparse_topk)
    core = HybridCore(cfg)
    ks = [int(x) for x in args.k.split(",")]
    train_rows = {k: LC.make_rows(args.train_rows, L=args.L, k=k, n_pairs=args.n_pairs, seed=0) for k in ks}
    held_rows = {k: LC.make_rows(args.held_rows, L=args.L, k=k, n_pairs=args.n_pairs, seed=1) for k in ks}
    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, args.warmup),
         optim.cosine_decay(args.lr, max(1, args.steps - args.warmup), args.lr * 0.05)], [args.warmup]),
        weight_decay=0.01)

    def lf(m, t, a, y):
        return nn.losses.cross_entropy(value_logits(m(t), a), y, reduction="mean")
    g = nn.value_and_grad(core, lf)
    rng = random.Random(10_000 + args.seed)
    log, t0, eval_s = [], time.time(), 0.0
    for it in range(1, args.steps + 1):
        rows = [train_rows[k][rng.randrange(len(train_rows[k]))] for k in (rng.choice(ks) for _ in range(args.batch))]
        loss, gr = g(core, *batch(rows))
        gr, _ = optim.clip_grad_norm(gr, 1.0)
        opt.update(core, gr)
        mx.eval(core.parameters(), opt.state)
        if it % args.log_every == 0 or it == args.steps:
            row = {"step": it, "loss": round(float(loss.item()), 4), "elapsed_s": round(time.time() - t0, 1),
                   "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3)}
            if it % (args.log_every * 5) == 0 or it == args.steps:
                te = time.time()
                row["held"] = {str(k): round(evaluate(core, held_rows[k][:256]), 4) for k in ks}
                eval_s += time.time() - te
            log.append(row)
            print(row, flush=True)
    wpath = out / "hybrid.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(core.parameters())))
    sha = hashlib.sha256(wpath.read_bytes()).hexdigest()
    name = f"ens-h40-hybrid-{args.pattern}-d{args.d}-L{args.L}-s{args.seed}@{sha[:12]}"
    receipt = {"schema": "ens.h40.hybrid_receipt.v0", "checkpoint": name, "config": cfg.to_json(), "args": vars(args),
               "params": sum(v.size for _, v in tree_flatten(core.parameters())),
               "held": {str(k): evaluate(core, held_rows[k]) for k in ks},
               "train_steps_per_s": round(args.steps / max(1e-9, time.time() - t0 - eval_s), 3),
               "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3), "chance": 1 / V.N_VALUES, "log": log,
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="GGGGA")
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--k", default="1", help="comma-separated hop counts, e.g. 1,2")
    p.add_argument("--n-pairs", type=int, default=48)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--sparse-block", type=int, default=32)
    p.add_argument("--sparse-topk", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--train-rows", type=int, default=20000)
    p.add_argument("--held-rows", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", required=True)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
