"""H41 v2: a bus-trained core that also learns in-context recall, so the endpoint depends on the cue.

v1 found that a C6 core never carries the cue K(x) to the answer endpoint: in bus training z is random, so x
never matters. Backward endpoint->evidence edges cannot work on such a core. This trainer keeps the C6 recipe
(bus rows with the jittered codebook message for z, plus ordinary rows) and adds recall rows, where z must be
looked up in context:

  [BOS  GR K(x1) V(t1)  GR K(x2) V(t2) ...  Q K(x) op.. SLOT NEUTRAL ANS]    t = T[x], answer = f_op(T[x])

T is a fresh random table (16 keys -> the family's z domain) per row, so nothing can be memorised; the queried
key is always among the facts. No bus call on recall rows. Families: the train-split PROGRAM families (z domain
16), so the A0 dev and confirm families stay unseen as recall data.

  python -m research.ens.recall_core --steps 6000 --out <dir>
"""
from __future__ import annotations

import argparse
import math
import random
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from research.ens import common as C
from research.ens import families as F
from research.ens import vocab as V
from research.ens.model import BusCall, Core, CoreConfig, value_logits
from research.ens.train_core import build_batch, family_eval, quick_eval, setup_trainable


def strip(ids):
    while ids and ids[-1] == V.PAD:
        ids = ids[:-1]
    return list(ids)


def recall_row(rng, fam, n_facts, T_len):
    """One recall row: (ids, query_idx, field_idx, ans_idx, answer, x, t)."""
    # a permutation: the facts in a row carry distinct values, so no non-lookup shortcut beats 1/n_facts (with
    # repeats, guessing the plurality fact value scored 0.31; review 2026-11-23)
    table = rng.sample(range(fam.z_domain), V.N_KEYS)
    keys = rng.sample(range(V.N_KEYS), n_facts)
    x = rng.choice(keys)
    row = F.sample_row(fam, rng, x=x, z=table[x])
    tail = strip(F.layout(row, "bus")[0])[1:]                    # drop BOS: [Q K(x) op.. SLOT NEUTRAL ANS]
    facts = [t for k in keys for t in (V.tok("GR"), V.key(k), V.val(table[k]))]
    ids = [V.BOS] + facts + tail
    if len(ids) > T_len:
        raise ValueError("recall row longer than seq_len")
    off = 1 + len(facts)
    qi, fi, ai = off + 1, len(ids) - 2, len(ids) - 1
    return ids + [V.PAD] * (T_len - len(ids)), qi, fi, ai, row.answer, x, table[x]


def recall_batch(rng, fams, n, n_facts, T_len):
    rows = [recall_row(rng, fams[rng.randrange(len(fams))], n_facts, T_len) for _ in range(n)]
    toks, q, f, a, ans = (mx.array([r[i] for r in rows]) for i in range(5))
    return toks, q, f, a, ans


def n_facts_at(it, args):
    """Curriculum (H41 v2b, 2026-11-23): the d256 48k-step run without one never learned in-context recall
    (0.07 at 5 facts). With --curriculum F, rows hold 1 fact for the first part of F x steps (copy the fact's value
    to the endpoint, no selection), then 2, 3, ... up to --n-facts by F x steps."""
    if args.curriculum <= 0:
        return args.n_facts
    assert args.curriculum_hold < args.curriculum <= 1, "need curriculum_hold < curriculum <= 1"
    hold, end = args.curriculum_hold * args.steps, args.curriculum * args.steps
    if it <= hold:
        return 1                                           # --curriculum-hold: a long copy-only phase first
    return min(args.n_facts, 1 + int((it - hold) / max(1.0, end - hold) * (args.n_facts - 1) + 1e-9))


def recall_acc(core, rng, fams, n, n_facts, T_len):
    toks, q, f, a, ans = recall_batch(rng, fams, n, n_facts, T_len)
    pred = mx.argmax(value_logits(core(toks), a), axis=-1)
    return float(mx.mean(pred == ans).item())


def train(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    cfg = CoreConfig(d=args.d, heads=args.heads, layers=args.layers, ffn=4 * args.d, bus_width=args.bus_width,
                     tau=args.tau)
    core = Core(cfg)
    site, M = args.site, C.codebook(cfg.bus_width)
    fams = F.split("train")
    rfams = [f for f in fams if f.cls == "PROGRAM"]
    data_rng, jit_rng, rec_rng = (random.Random(k + args.seed) for k in (10_000, 30_000, 50_000))
    setup_trainable(core, "c6", site)                   # as C6: backbone + the site port (tap frozen); other ports frozen
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, args.warmup),
         optim.cosine_decay(args.lr, max(1, args.steps - args.warmup), args.lr * 0.05)], [args.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)

    def loss_fn(model, tokens, q, f, a, ans, msg):
        bus = None if msg is None else BusCall(site, q, f, msg)
        return nn.losses.cross_entropy(value_logits(model(tokens, bus), a), ans, reduction="mean")
    grad_fn = nn.value_and_grad(core, loss_fn)
    state = [core.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_bus(tokens, q, f, a, ans, msg):
        loss, g = grad_fn(core, tokens, q, f, a, ans, msg)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(core, g)
        return loss

    @partial(mx.compile, inputs=state, outputs=state)
    def step_plain(tokens, q, f, a, ans):
        loss, g = grad_fn(core, tokens, q, f, a, ans, None)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(core, g)
        return loss

    log, t0 = [], time.time()
    for it in range(1, args.steps + 1):
        if data_rng.random() < args.p_recall:
            kind = "recall"
            loss = step_plain(*recall_batch(rec_rng, rfams, args.batch, n_facts_at(it, args), cfg.seq_len))
        else:
            kind = "c6"
            b = build_batch(data_rng, jit_rng, fams, args.batch, "c6", args.p_ordinary, M)
            loss = step_bus(*b)
        mx.eval(state)
        if it % args.log_every == 0 or it == args.steps:
            lv = float(loss.item())
            if not math.isfinite(lv):
                raise SystemExit(f"non-finite loss at step {it}")
            row = {"step": it, "loss": round(lv, 5), "batch_kind": kind, "elapsed_s": round(time.time() - t0, 1),
                   "recall_acc": recall_acc(core, random.Random(99), rfams, 512, args.n_facts, cfg.seq_len),
                   "recall_acc_1fact": recall_acc(core, random.Random(98), rfams, 512, 1, cfg.seq_len),
                   "n_facts_now": n_facts_at(it, args)}
            if it == args.steps or it % (args.log_every * args.eval_every) == 0:
                row.update(quick_eval(core, "c6", M, cfg, site))
                if row.get("acc_ordinary"):                 # recall over the op skill alone (ordinary rows: z given)
                    row["recall_over_ordinary"] = round(row["recall_acc"] / row["acc_ordinary"], 4)
                    row["recall1_over_ordinary"] = round(row["recall_acc_1fact"] / row["acc_ordinary"], 4)
            log.append(row)
            print(row, flush=True)
            C.write_json(out / "progress.json", {"log": log, "args": vars(args)})   # kept even if a cap stops the run
    fam_eval = family_eval(core, "c6", M, site)          # before saving: a crash here must not leave weights alone
    name = f"ens-h41-core-c6recall-{site}-w{cfg.bus_width}-d{cfg.d}-L{cfg.layers}-s{args.seed}"
    ckpt = C.save_core(core, out, name)
    receipt = {"recall_chance": round(1 / args.n_facts, 4), "p_ordinary_effective": round((1 - args.p_recall) * args.p_ordinary, 4),"schema": "ens.h41.recall_core_receipt.v0", "checkpoint": ckpt, "arm": "c6recall", "seed": args.seed,
               "site": site, "config": cfg.to_json(), "args": vars(args), "log": log,
               "families": F.registry_record(), "final_family_eval": fam_eval,
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--bus-width", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--site", default="early")
    p.add_argument("--p-ordinary", type=float, default=0.25)
    p.add_argument("--p-recall", type=float, default=0.33)
    p.add_argument("--n-facts", type=int, default=5)
    p.add_argument("--curriculum", type=float, default=0.0, help="reach --n-facts by this fraction of steps")
    p.add_argument("--curriculum-hold", type=float, default=0.0, help="1 fact only for this fraction of steps first")
    p.add_argument("--eval-every", type=int, default=4, help="quick_eval every this many log points")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
