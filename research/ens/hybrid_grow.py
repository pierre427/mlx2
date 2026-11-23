"""H40 x growth (Pierre, 2026-11-23): a GDN-hybrid seed "GGGA" (three gated delta-net blocks and one dense RoPE
attention block, the bus after block 1) that grows by whole groups: append another "GGGA" (born as the identity)
when needed, to "GGGAGGGA", and further if needed.

Trained on the A0 C6 recipe (research.ens.train_core.build_batch: bus rows carry the jittered codebook message for
z at the field position, plus ordinary rows), seq_len 32. The growth operator is exact: every block is residual,
h + o(...) then h + down(...), so a new block with zero o and zero FFN down projections is the identity, and
appending after the last block keeps the bus site. Adam moments are mapped (old entries kept, new ones zero).

Arms, compared at equal training FLOPs (6 * params * tokens per step):
  grow     --pattern GGGA --grow-at 0.25 [--grow-group GGGA]   append a group at that fraction of the steps
  fixed    --pattern GGGA or --pattern GGGAGGGA                 no growth

  python -m research.ens.hybrid_grow --pattern GGGA --grow-at 0.25 --steps 24000 --out <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from research.ens import common as C
from research.ens import families as F
from research.ens.hybrid import HybridConfig, HybridCore
from research.ens.model import BusCall
from research.ens.train_core import build_batch


def n_params(m):
    return sum(v.size for _, v in tree_flatten(m.parameters()))


def append_group(core: HybridCore, group: str, *, moments: bool = False) -> HybridCore:
    """Append the blocks of `group` after the last block, each born as the identity (o and ffn.down zero).
    With moments=True map an optimiser moment tree instead (old entries kept, every new entry zero)."""
    cfg = core.cfg
    new = HybridCore(dataclasses.replace(cfg, pattern=cfg.pattern + group))
    old = dict(tree_flatten(core.parameters()))
    out = []
    n0 = len(cfg.pattern)
    for name, nv in tree_flatten(new.parameters()):
        parts = name.split(".")
        if parts[0] == "blocks" and int(parts[1]) >= n0:
            zero = moments or parts[2] == "o" or (parts[2] == "ffn" and parts[3] == "down")
            out.append((name, mx.zeros_like(nv) if zero else nv))
        else:
            ov = old[name]
            if ov.shape != nv.shape:
                raise ValueError(f"append_group: shape changed for {name}")
            out.append((name, ov))
    new.load_weights(out)
    return new


def map_optimizer(core, opt, grown, group, make_opt):
    flat = dict(tree_flatten(opt.state))
    trees = {}
    for k in ("m", "v"):
        src = HybridCore(core.cfg)
        src.load_weights([(n, flat[f"{n}.{k}"]) for n, _ in tree_flatten(core.parameters())])
        trees[k] = dict(tree_flatten(append_group(src, group, moments=True).parameters()))
    new = make_opt()
    new.init(grown.trainable_parameters())
    state = tree_unflatten([(f"{n}.{k}", trees[k][n]) for n, _ in tree_flatten(grown.parameters()) for k in ("m", "v")])
    state["step"] = opt.state["step"]
    state["learning_rate"] = new.state["learning_rate"]
    new.state = state
    return new


def lr_at(step, args, grew_at=None):
    """Warmup then cosine. After a growth at step g, a linear re-warmup from 10% of the scheduled rate over
    --rewarmup steps (review 2026-11-23): the new blocks' AdamW moments start at zero without bias correction, so
    their first updates would be several times the rate, at near-peak rate. Applied to every parameter, so it
    also slows the old blocks briefly; recorded in the receipt."""
    if step < args.warmup:
        base = args.lr * step / args.warmup
    else:
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        base = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))
    if grew_at is not None and args.rewarmup > 0 and step - grew_at < args.rewarmup:
        base *= 0.1 + 0.9 * (step - grew_at) / args.rewarmup
    return base


def vlogits(logits, a):
    from research.ens import vocab as V
    B = logits.shape[0]
    return logits[mx.arange(B), a][:, V.V0:V.V0 + V.N_VALUES]


def evaluate(core, batch_):
    tokens, q, f, a, ans, msg = batch_
    bus = BusCall("early", q, f, msg) if msg is not None else None
    pred = mx.argmax(vlogits(core(tokens, bus), a), axis=-1)
    return float(mx.mean(pred == ans).item())


def train(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    cfg = HybridConfig(d=args.d, heads=args.heads, ffn=4 * args.d, seq_len=32, pattern=args.pattern,
                       bus_width=args.bus_width, tau=args.tau, site=args.site, gdn_chunk=args.gdn_chunk)
    core = HybridCore(cfg)
    M = C.codebook(cfg.bus_width)
    fams = F.split("train")
    data_rng, jit_rng = random.Random(10_000 + args.seed), random.Random(30_000 + args.seed)
    val = build_batch(random.Random(77), random.Random(78), fams, 1024, "c6", args.p_ordinary, M)
    # held evaluation: bus rows only and ordinary rows only, separately
    ev_bus = build_batch(random.Random(87), random.Random(88), fams, 1024, "c6", 0.0, M)
    ev_ord = build_batch(random.Random(97), random.Random(98), fams, 1024, "c6", 1.0, M)

    def loss_fn(model, tokens, q, f, a, ans, msg):
        return nn.losses.cross_entropy(vlogits(model(tokens, BusCall("early", q, f, msg)), a), ans, reduction="mean")

    def make_opt():
        return optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    opt = make_opt()
    flops, log, events, t0 = 0, [], [], time.time()
    grow_step = int(args.grow_at * args.steps) if args.grow_at > 0 else -1
    if args.grow_at > 0 and not 1 <= grow_step <= args.steps:
        raise SystemExit(f"--grow-at {args.grow_at} gives growth step {grow_step}, outside 1..{args.steps}")
    grew_at = None
    g = nn.value_and_grad(core, loss_fn)
    for it in range(1, args.steps + 1):
        if it == grow_step:
            grown = append_group(core, args.grow_group)
            opt = map_optimizer(core, opt, grown, args.grow_group, make_opt)
            core = grown
            grew_at = it
            g = nn.value_and_grad(core, loss_fn)
            events.append({"step": it, "op": f"append_group:{args.grow_group}", "pattern": core.cfg.pattern,
                           "params_after": n_params(core)})
            print("growth", events[-1], flush=True)
        opt.learning_rate = lr_at(it, args, grew_at)
        b = build_batch(data_rng, jit_rng, fams, args.batch, "c6", args.p_ordinary, M)
        loss, gr = g(core, *b)
        gr, _ = optim.clip_grad_norm(gr, 1.0)
        opt.update(core, gr)
        mx.eval(core.parameters(), opt.state)
        flops += 6 * n_params(core) * int(b[0].size)
        if it % args.log_every == 0 or it == args.steps:
            lv = float(loss.item())
            if not math.isfinite(lv):
                raise SystemExit(f"non-finite loss at step {it}")
            row = {"step": it, "pattern": core.cfg.pattern, "params": n_params(core), "train_flops": flops,
                   "val_loss": round(float(loss_fn(core, *val).item()), 5), "acc_bus": evaluate(core, ev_bus),
                   "acc_ordinary": evaluate(core, ev_ord), "elapsed_s": round(time.time() - t0, 1)}
            log.append(row)
            print(row, flush=True)
            C.write_json(out / "progress.json", {"log": log, "events": events})
    wpath = out / "core.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(core.parameters())))
    with wpath.open("rb") as fh:
        fp = hashlib.file_digest(fh, "sha256").hexdigest()[:12]
    name = f"ens-h40-hybrid-{args.pattern}-{core.cfg.pattern}-d{cfg.d}-s{args.seed}"
    elapsed = time.time() - t0
    receipt = {"schema": "ens.h40.hybrid_grow_receipt.v0", "checkpoint": f"{name}@{fp}", "pattern_start": args.pattern,
               "codebook_sha256": C.codebook_digest(cfg.bus_width), "steps_per_s": round(args.steps / elapsed, 2),
               "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3), "rewarmup_after_growth": args.rewarmup,
               "config": core.cfg.to_json(), "args": vars(args), "params": n_params(core), "train_flops": flops,
               "growth_events": events, "log": log,
               "families": F.registry_record(), **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="GGGA")
    p.add_argument("--grow-at", type=float, default=0.0, help="append --grow-group at this fraction of the steps")
    p.add_argument("--grow-group", default="GGGA")
    p.add_argument("--rewarmup", type=int, default=400, help="LR re-warmup steps after a growth (10%% -> 100%%)")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--bus-width", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--site", type=int, default=1)
    p.add_argument("--gdn-chunk", type=int, default=16)
    p.add_argument("--p-ordinary", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=800)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
