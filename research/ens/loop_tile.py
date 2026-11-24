"""H45 client 1: a halting tile on a frozen loop core (the port's stop signal).

After every pass the tile reads the core's final-norm state at the answer position and says halt or go on. It is
trained (BCE) on the core-training mix with the oracle stop pass = the row's step count (depths 1-3, passes up to
4, so pass 4 is always a 'go on too long' negative). At evaluation the core runs until the tile halts or the budget
(8 passes) is spent. Compared with the oracle pass count on every depth family, trained (1-3) and unseen (4-6).

Kill measure (client level): on trained depths, tile-driven bus accuracy more than 0.05 below oracle-count
accuracy. Unseen depths are reported: halting there needs the state to carry how many steps remain.

  python -m research.ens.loop_tile --core <loop core dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import families as F
from research.ens import loop_train as L
from research.ens.model import BusCall, value_logits

TRAIN_PASSES = L.TRAIN_MAX_PASSES + 1
BUDGET = 8


class Halt(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.l1 = nn.Linear(d, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def __call__(self, s):
        return self.l2(nn.silu(self.l1(s)))[..., 0]


def states_at_answer(core, tokens, q, f, a, msg, passes_max):
    """(P, B, d) final-norm states at the answer position after passes 1..P."""
    B = tokens.shape[0]
    _, st = core.looped(tokens, mx.full((B,), passes_max), BusCall("early", q, f, msg), return_states=True)
    return st[:, mx.arange(B), a]


def run(args):
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    if tuple(core.cfg.loop_body) != L.LOOP_BODY:
        raise SystemExit("not an H45 loop core")
    core.freeze()
    M = C.codebook(core.cfg.bus_width)
    mx.random.seed(args.seed)
    tile = Halt(core.cfg.d)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    fams = L.train_mix()

    def lf(m, st, target):                               # st (P, B, d); target (P, B) 1 at the stop pass
        return nn.losses.binary_cross_entropy(m(st), target, with_logits=True, reduction="mean")
    g = nn.value_and_grad(tile, lf)
    rng, jrng = random.Random(60_000 + args.seed), random.Random(61_000 + args.seed)
    log, t0 = [], time.time()
    for it in range(1, args.steps + 1):
        tokens, q, f, a, ans, msg, passes = L.build_batch(rng, jrng, fams, args.batch, 0.0, M)
        st = mx.stop_gradient(states_at_answer(core, tokens, q, f, a, msg, TRAIN_PASSES))
        target = (mx.arange(1, TRAIN_PASSES + 1)[:, None] == passes[None, :]).astype(mx.float32)
        loss, gr = g(tile, st, target)
        opt.update(tile, gr)
        mx.eval(tile.parameters(), opt.state)
        if it % args.log_every == 0 or it == args.steps:
            log.append({"step": it, "loss": round(float(loss.item()), 5), "elapsed_s": round(time.time() - t0, 1)})
            print(log[-1], flush=True)
    tile.freeze()
    mx.save_safetensors(str(out / "halt.safetensors"), dict(tree_flatten(tile.parameters())))
    res = {}
    for fam in L.depth_families():
        r = random.Random(int(hashlib.sha256(f"h45.depth_eval:{fam.fid}".encode()).hexdigest()[:8], 16))
        rows = [F.sample_row(fam, r) for _ in range(args.eval_n)]
        s = L.steps_of(fam)
        tokens, q, f, a, ans, zs = C.batch(rows, "bus")
        B = len(rows)
        _, st = core.looped(tokens, mx.full((B,), BUDGET), BusCall("early", q, f, M[zs]), return_states=True)
        halt = tile(st[:, mx.arange(B), a]) > 0                       # (BUDGET, B)
        first = mx.where(mx.any(halt, axis=0), mx.argmax(halt.astype(mx.int32), axis=0) + 1, BUDGET)
        logits_all = core.backbone.head(st)                            # (BUDGET, B, T, V)
        pred = mx.argmax(value_logits(logits_all[first - 1, mx.arange(B)], a), axis=-1)
        oracle = mx.argmax(value_logits(logits_all[s - 1, mx.arange(B)], a), axis=-1) if s <= BUDGET else None
        res[fam.fid] = {"steps": s, "trained_depth": s <= L.TRAIN_MAX_PASSES,
                        "halt_exact": round(float(mx.mean(first == s).item()), 4),
                        "mean_passes": round(float(mx.mean(first.astype(mx.float32)).item()), 3),
                        "acc_tile": round(float(mx.mean(pred == ans).item()), 4),
                        "acc_oracle_count": round(float(mx.mean(oracle == ans).item()), 4)}

    def mean(key, trained):
        v = [x[key] for x in res.values() if x["trained_depth"] == trained]
        return round(sum(v) / len(v), 4)
    summary = {k: {"trained": mean(k, True), "unseen": mean(k, False)}
               for k in ("halt_exact", "acc_tile", "acc_oracle_count")}
    summary["killed"] = summary["acc_tile"]["trained"] < summary["acc_oracle_count"]["trained"] - 0.05
    receipt = {"schema": "ens.h45.halt_tile_receipt.v1", "core_checkpoint": ckpt, "seed": args.seed,
               "budget": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "train_passes": TRAIN_PASSES,
                          "eval_budget": BUDGET, "eval_n": args.eval_n},
               "log": log, "depth": res, "summary": summary, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print("summary", json.dumps(summary), flush=True)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--log-every", type=int, default=250)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
