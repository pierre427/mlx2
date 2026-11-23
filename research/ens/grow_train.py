"""Grow a bus-trained core during training (H33 / H40 groundwork): start small, grow on stalls, earned on validation.

The training data and loss are the C6 recipe (research.ens.train_core.build_batch: bus rows carry the jittered
codebook message for a random z, plus ordinary rows). Every --window steps the controller measures validation loss
on a fixed held batch. If the relative improvement over the last window is below --stall, it runs a growth window:

  candidates   insert_block (middle of the stack), widen_ffn (x1.5), widen_d (+1 head); all function-preserving
  trial        each candidate AND a no-growth control are trained --trial-steps on copies with the same batches
  accept       the candidate with the best validation gain over the control, if that gain > --margin (earned growth)

Budget (carrying capacity, training plan section 12): the kept arm's trial steps count against --steps, the other
arms' trial steps stop once they reach --trial-budget of all steps, and no growth beyond --max-params. Every growth window is written to the receipt.
The control continues with the run's own AdamW; a grown candidate continues with those AdamW moments mapped through
its growth operator (new entries 0; --fresh-opt restarts AdamW instead, the earlier behaviour). Trial steps of rejected arms are extra compute (trial_steps_used).

  python -m research.ens.grow_train --d 128 --heads 2 --layers 4 --steps 48000 --out <dir>
"""
from __future__ import annotations

import argparse
import math
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import families as F
from research.ens import growth as G
from research.ens.model import BusCall, Core, CoreConfig, value_logits
from research.ens.train_core import build_batch, family_eval, quick_eval


def n_params(core):
    return sum(v.size for _, v in tree_flatten(core.parameters()))


def make_loss(site):
    def loss_fn(model, tokens, q, f, a, ans, msg):
        return nn.losses.cross_entropy(value_logits(model(tokens, BusCall(site, q, f, msg)), a), ans, reduction="mean")
    return loss_fn


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * step / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def new_opt(step, args):
    return optim.AdamW(learning_rate=lr_at(step, args), weight_decay=0.01)


def step_flops(core, tokens):
    """Training FLOPs of one step: the dense 6 * params * tokens estimate (forward 2, backward 4), tokens counted
    at the padded seq_len (the model computes every position). Two small biases, a few percent at d 128 and
    partly cancelling (review 2026-11-22): N includes parameters that do no matmul work (embeddings, unused
    ports; overstates small models, against growth), and the attention-score term (about T / 8d of the dense term)
    is left out (understates small models, for growth). Shapes differ between arms, so neither is constant."""
    return 6 * n_params(core) * int(tokens.size)


def unit_importance(core, gr):
    """First-order Taylor importance of one step (H42). Unit j of a block: |sum(w_down[:, j] * g_down[:, j])|, which
    equals the first-order loss change of zeroing the unit (sum over tokens of dL/dm_j * m_j; the unit is bilinear
    in up and down, so adding the up or gate rows would double-count or mix in a different term; review
    2026-11-22). Block: |sum(w * g)| over its output projections (o, down), the change from zeroing the block to
    the identity. Gradients before clipping: clipping is a global per-step scale."""
    F0 = core.cfg.ffn
    units, blocks = [], []
    for blk, gb in zip(core.backbone.blocks, gr["backbone"]["blocks"]):
        wd, gd = blk.down.weight, gb["down"]["weight"]
        units.append(mx.abs(mx.sum(wd * gd, axis=0)))
        blocks.append(mx.abs(mx.sum(blk.o.weight * gb["o"]["weight"]) + mx.sum(wd * gd)))
    return mx.stack(units), mx.stack(blocks)


def train_steps(core, batches, start_step, args, site, opt, meter=None, imp=None):
    """Train in place with the given optimiser (kept across windows; replaced only when the core grows).
    meter: a dict whose "flops" entry accumulates the training FLOPs of every step run here.
    imp: a dict holding sliding (EMA) importance "units" (layers, ffn) and "blocks" (layers,), updated each step."""
    if meter is not None and batches:
        meter["flops"] = meter.get("flops", 0) + len(batches) * step_flops(core, batches[0][0])
    g = nn.value_and_grad(core, make_loss(site))
    for i, b in enumerate(batches):
        opt.learning_rate = lr_at(start_step + i, args)
        _, gr = g(core, *b)
        if imp is not None:
            u, bl = unit_importance(core, gr)
            if imp.get("units") is None or imp["units"].shape != u.shape:
                imp["units"], imp["blocks"], imp["n"] = u, bl, 0
            else:
                beta = args.ema
                imp["units"] = beta * imp["units"] + (1 - beta) * u
                imp["blocks"] = beta * imp["blocks"] + (1 - beta) * bl
            imp["n"] += 1
        gr, _ = optim.clip_grad_norm(gr, 1.0)
        opt.update(core, gr)
        # evaluate the importance every step: otherwise the EMA chains a lazy graph that keeps every step's weights
        # and gradients alive until the next prune decision (review 2026-11-22: about 6.8 MB per step at d 128)
        mx.eval(core.parameters(), opt.state, *([imp["units"], imp["blocks"]] if imp else []))
    return core


def balloon_window(core, opt, imp, step, windows, P0, args, site, imp_state):
    """H42 (Pierre, 2026-11-22): balloon early, prune on sliding windows throughout.
    Balloon phase (step < --balloon-until x steps): on even windows, grow without trials, alternating insert_block
    (first, at window 2) and widen_ffn (x --grow-ffn-mult), while the core stays under --balloon-cap x the start
    params. Pruning (odd windows during the balloon phase, every window after it), once the sliding importance has
    seen a full window since the last shape change and at least --prune-grace windows have passed since the last
    growth (new units start with zero down columns, so they score zero at birth; review 2026-11-22):
      units   remove the weak FFN units: per block, those under --unit-thresh x the block median; the number
              removed is the MINIMUM weak count over blocks (the FFN width is shared, so no block loses a strong
              unit), at most --prune-frac of the width per window, never below --min-ffn;
      blocks  remove the weakest block after the bus site if its importance is under --block-thresh x the mean
              block importance (never below --min-layers).
    Adam moments are mapped through every operator."""
    ev, ops = {}, []
    balloon = step < args.balloon_until * args.steps
    if balloon and windows % 2 == 0:
        grow_ffn = (windows // 2) % 2 == 0
        op, a = (G.widen_ffn, (int(core.cfg.ffn * args.grow_ffn_mult),)) if grow_ffn else \
            (G.insert_block, (core.cfg.layers // 2 + 1,))
        mx.random.seed(args.seed * 1_000_003 + step * 7)
        c = op(core, *a)
        if n_params(c) <= args.balloon_cap * P0 and n_params(c) <= args.max_params:
            ops.append((op, a, c, "grow:" + op.__name__))
    elif imp and imp.get("n", 0) >= args.window and windows - imp_state["last_grow"] > args.prune_grace:
        units, blocks = imp["units"], imp["blocks"]
        # harvest only the weak: units under --unit-thresh x their block's median importance, at most --prune-frac
        # of the units per window (the mean weak count over blocks, since the FFN width is shared)
        med = mx.sort(units, axis=1)[:, units.shape[1] // 2:units.shape[1] // 2 + 1]
        n_weak = int(mx.min(mx.sum(units < args.unit_thresh * med, axis=1)).item())
        n_rm = min(int(args.prune_frac * core.cfg.ffn), n_weak)
        k = max(args.min_ffn, core.cfg.ffn - n_rm)
        ev["weak_units_min"] = n_weak
        if k < core.cfg.ffn:
            order = mx.argsort(-units, axis=1)[:, :k].tolist()
            ops.append((G.prune_ffn, (order,), G.prune_ffn(core, order), f"prune_ffn:{core.cfg.ffn}->{k}"))
        first = dict(core.cfg.sites)[site] + 1                    # never remove a block at or before the bus site
        cand = list(range(first, core.cfg.layers))
        if cand and core.cfg.layers > args.min_layers:
            bl = blocks.tolist()
            weakest = min(cand, key=lambda i: bl[i])
            if bl[weakest] < args.block_thresh * (sum(bl) / len(bl)):
                ops.append((G.remove_block, (weakest,), None, f"remove_block:{weakest}"))
        ev["importance_blocks"] = [round(x, 6) for x in blocks.tolist()]
    for op, a, c, label in ops:
        c = c if c is not None else op(core, *a)
        opt = G.map_optimizer(op, core, opt, c, lambda: new_opt(step, args), *a)
        core = c
        ev.setdefault("ops", []).append(label)
        if label.startswith("grow:"):
            imp_state["last_grow"] = windows
    if ops:
        imp.clear()                                               # importance restarts at the new shape
    return core, opt, (ev if ev.get("ops") else {})


def decide(v, res, params, rule, margin):
    """Choose a growth candidate from trial results. v: the pre-trial selection loss (every arm starts from it, the
    operators being function-preserving); res: trial loss per arm, "control" = the true continuation; params: per
    arm. Returns (best, gain, accepted).

    loss      best = lowest trial loss; gain = control loss - best loss.
    per-flop  growth must buy more loss than continuing would at the candidate's per-step cost r = P_k / P_control
              (the 6*N*T rule; T cancels): s_k = (v - res_k) - r_k * max(v - res_control, 0). Multiplying, not
              dividing, and clamping the control's improvement at zero: when the control regresses, a candidate
              must improve in absolute nats, so a worse, larger model is never preferred (review 2026-11-22: the
              ratio form rewarded size whenever improvements were negative).
    accepted  gain > margin (nats)."""
    others = [k for k in res if k != "control"]
    if not others:
        return None, 0.0, False
    if rule == "per-flop":
        dc = max(v - res["control"], 0.0)
        score = {k: (v - res[k]) - params[k] / params["control"] * dc for k in others}
        best = max(others, key=score.get)
        gain = score[best]
    else:
        best = min(others, key=res.get)
        gain = res["control"] - res[best]
    return best, gain, bool(gain > margin)


def val_loss(core, val, site):
    return float(make_loss(site)(core, *val).item())


def train(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    cfg = CoreConfig(d=args.d, heads=args.heads, layers=args.layers, ffn=args.ffn or 4 * args.d, bus_width=args.bus_width,
                     tau=args.tau)
    core = Core(cfg)
    site = args.site
    M = C.codebook(cfg.bus_width)
    fams = F.split("train")
    data_rng, jit_rng = random.Random(10_000 + args.seed), random.Random(30_000 + args.seed)
    # selection batch (stall detection and accept decisions) and a separate report batch (logged val_loss), so the
    # reported curve carries no winner's-curse bias from choosing among candidates on the same rows
    val = build_batch(random.Random(77), random.Random(78), fams, 1024, "c6", args.p_ordinary, M)
    report = build_batch(random.Random(87), random.Random(88), fams, 1024, "c6", args.p_ordinary, M)
    next_batch = lambda: build_batch(data_rng, jit_rng, fams, args.batch, "c6", args.p_ordinary, M)
    step, trial_used, log, events, windows = 0, 0, [], [], 0
    meter = {"flops": 0}                 # every training step, kept arm and rejected trial arms alike
    last_val, t0 = val_loss(core, val, site), time.time()
    trial_cap = int(args.trial_budget * args.steps)
    opt = new_opt(0, args)
    P0 = n_params(core)
    imp = {} if args.mode == "balloon" else None
    imp_state = {"last_grow": -10**9}
    while step < args.steps:
        n = min(args.window, args.steps - step)
        core = train_steps(core, [next_batch() for _ in range(n)], step, args, site, opt, meter, imp)
        step += n
        windows += 1
        v = val_loss(core, val, site)
        row = {"step": step, "window": windows, "val_loss": round(val_loss(core, report, site), 5), "sel_loss": round(v, 5), "params": n_params(core), "shape": [core.cfg.layers, core.cfg.d,
               core.cfg.ffn, core.cfg.heads], "train_flops": meter["flops"], "elapsed_s": round(time.time() - t0, 1)}
        if windows % args.eval_every == 0 or step >= args.steps:      # by window count: growth shifts step
            row.update(quick_eval(core, "c6", M, core.cfg, site))
            row["families"] = family_eval(core, "c6", M, site)
        log.append(row)
        print({k: row[k] for k in row if k != "families"}, flush=True)
        rel = (last_val - v) / max(abs(last_val), 1e-9)
        last_val = v
        stalled = rel < args.stall
        if args.mode == "balloon" and step < args.steps:
            core, opt, ev = balloon_window(core, opt, imp, step, windows, P0, args, site, imp_state)
            if ev:
                ev.update({"step": step, "params_after": n_params(core)})
                events.append(ev)
                print("balloon window", ev, flush=True)
            continue
        # a growth window must end before --steps, so the run always ends on a logged, evaluated window (review)
        if args.no_grow or not stalled or step + args.trial_steps >= args.steps:
            continue
        # growth window: candidates and a no-growth control see the same batches. The control is the true
        # continuation (the current model with its own optimiser state); a grown candidate continues with the same
        # state mapped through its operator. Growth must beat genuine continuation to be accepted.
        mid = core.cfg.layers // 2 + 1
        ops = {f"insert_block@{mid}": (G.insert_block, (mid,)), "widen_ffn": (G.widen_ffn, (int(core.cfg.ffn * 1.5),)),
               "widen_d+1head": (G.widen_d, (1,))}
        cands, opts = {"control": core}, {"control": opt}
        for j, (k, (op, a)) in enumerate(ops.items()):
            # reseed per (seed, step, candidate): a candidate's random fills do not depend on what ran before it
            # (mapping builds scratch cores), so mapped and --fresh-opt runs are a same-seed A/B (review 2026-11-22)
            mx.random.seed(args.seed * 1_000_003 + step * 7 + j)
            c = op(core, *a)
            if n_params(c) > args.max_params:
                continue
            cands[k] = c
            # the grown arm keeps the run's optimiser state, mapped through the same operator (--fresh-opt: restart)
            opts[k] = new_opt(step, args) if args.fresh_opt else G.map_optimizer(op, core, opt, c, lambda: new_opt(step, args), *a)
        if trial_used + args.trial_steps * (len(cands) - 1) > trial_cap or len(cands) == 1:
            continue
        batches = [next_batch() for _ in range(args.trial_steps)]   # drawn only once the window will run
        res = {}
        for k, c in cands.items():
            cands[k] = train_steps(c, batches, step, args, site, opts[k], meter)
            res[k] = val_loss(cands[k], val, site)
        trial_used += args.trial_steps * (len(cands) - 1)   # extra steps: the control's are ordinary training
        best, gain, accepted = decide(v, res, {k: n_params(c) for k, c in cands.items()}, args.accept, args.margin)
        ev = {"step": step, "rel_improvement": round(rel, 5), "trial_val": {k: round(x, 5) for k, x in res.items()},
              "accept_rule": args.accept, "pre_trial_sel": round(v, 5),
              "best": best, "gain_over_control": round(gain, 5), "accepted": accepted}
        keep = best if ev["accepted"] else "control"
        core, opt = cands[keep], opts[keep]
        step += args.trial_steps                         # the kept model trained these steps; count them
        ev["params_after"] = n_params(core)
        ev["trial_params"] = {k: n_params(c) for k, c in cands.items()}
        events.append(ev)
        print("growth window", ev, flush=True)
    c = core.cfg
    name = f"ens-a0-core-c6grow-{site}-w{c.bus_width}-d{c.d}-L{c.layers}-f{c.ffn}-h{c.heads}-s{args.seed}"
    ckpt = C.save_core(core, out, name)
    receipt = {"schema": "ens.a0.grow_receipt.v0", "checkpoint": ckpt, "arm": "c6", "seed": args.seed, "site": site,
               "config": core.cfg.to_json(), "start_config": cfg.to_json(), "args": vars(args),
               "params": n_params(core), "growth_events": events, "trial_steps_used": trial_used, "train_flops": meter["flops"], "flops_rule": "6*params*tokens per step, all arms",
               "trial_budget_steps": trial_cap, "log": log, "families": F.registry_record(),
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=0, help="FFN width (default 4*d)")
    p.add_argument("--bus-width", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--site", default="early")
    p.add_argument("--p-ordinary", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=48000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--stall", type=float, default=0.01, help="relative val improvement per window below which to try growth")
    p.add_argument("--trial-steps", type=int, default=200)
    p.add_argument("--trial-budget", type=float, default=0.10)
    p.add_argument("--margin", type=float, default=0.005)
    p.add_argument("--max-params", type=int, default=12_000_000)
    p.add_argument("--mode", default="earned", choices=("earned", "balloon"),
                   help="earned: trial-gated growth on stalls; balloon: H42, scheduled growth early plus "
                        "sliding-window importance pruning throughout")
    p.add_argument("--balloon-cap", type=float, default=4.0, help="balloon: max params as a multiple of the start")
    p.add_argument("--balloon-until", type=float, default=0.3, help="balloon: growth allowed before this fraction of steps")
    p.add_argument("--prune-frac", type=float, default=0.1, help="balloon: FFN units removed per prune window")
    p.add_argument("--min-ffn", type=int, default=32)
    p.add_argument("--unit-thresh", type=float, default=0.25, help="balloon: a unit is weak under this x its block median")
    p.add_argument("--grow-ffn-mult", type=float, default=2.0, help="balloon: FFN width multiplier per growth")
    p.add_argument("--min-layers", type=int, default=2)
    p.add_argument("--block-thresh", type=float, default=0.25)
    p.add_argument("--prune-grace", type=int, default=2, help="balloon: windows after a growth before pruning")
    p.add_argument("--ema", type=float, default=0.98, help="balloon: per-step decay of the sliding importance")
    p.add_argument("--accept", default="loss", choices=("loss", "per-flop"),
                   help="loss: best trial loss beats the continuation by --margin; per-flop: the candidate's trial "
                        "gain minus its cost ratio times the continuation's gain exceeds --margin (see decide)")
    p.add_argument("--fresh-opt", action="store_true", help="grown arms restart AdamW instead of mapping its state")
    p.add_argument("--no-grow", action="store_true", help="fixed-shape control arm (same loop, no growth windows)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
