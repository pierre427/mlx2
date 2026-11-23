"""Train one A0 core arm (spec v0.4 section 6.1, step 2).

  c6   bus-trained: the oracle mock provider writes M[z] (+ ABI jitter) through the
       bus site (--site, default early); the typed field holds NEUTRAL.
  c7   typed-field control: z arrives as an ordinary typed token at the same
       position; the bus is present (frozen scaled nonzero reader, open gate)
       but carries nothing.
  c6r  secondary: the backbone of a finished c7 run, frozen; only the bus-site
       reader and gate train on bus episodes.

Both primary arms share initialisation (seeded by --seed), data order, optimizer
and update count, and see the same usable information per row. A fraction of
every batch is 'ordinary' rows (z given in the prompt, no provider) so the
ordinary-task regression epsilon can be measured on the same core.

  python -m research.ens.train_core --arm c6 --seed 0 --steps 20000 --out <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import families as F
from research.ens.model import BusCall, Core, CoreConfig, value_logits

DEFAULT_SITE = "early"


def make_rows(rng: random.Random, fams, n):
    return [F.sample_row(fams[rng.randrange(len(fams))], rng) for _ in range(n)]


def build_batch(rng, jrng, fams, n, arm, p_ordinary, M, matryoshka=0.0):
    """Return (tokens, qidx, fidx, aidx, answers, message-or-None).

    rng drives rows and modes identically in every arm; jitter draws come from a separate jrng
    so the bus arms do not shift the shared data order (review finding, 2026-11-21).
    """
    rows = make_rows(rng, fams, n)
    n_ord = int(round(p_ordinary * n))
    modes = ["ordinary"] * n_ord + (["typed"] if arm == "c7" else ["bus"]) * (n - n_ord)
    rng.shuffle(modes)
    toks, q, f, a, ans, zs = [], [], [], [], [], []
    for r, m in zip(rows, modes):
        ids, qi, fi, ai = F.layout(r, m)
        toks.append(ids); q.append(qi); f.append(fi); a.append(ai); ans.append(r.answer)
    tokens, q, f, a, ans = (mx.array(x) for x in (toks, q, f, a, ans))
    if arm == "c7":
        return tokens, q, f, a, ans, None
    z = mx.array([r.z for r in rows])
    has = mx.array([m == "bus" for m in modes])[:, None]
    msg = C.jitter(M[z], jrng) * has                     # ordinary rows carry a zero message
    if matryoshka > 0:                                   # H26: some bus rows keep only the low half of the bus
        w = msg.shape[1]
        cut = mx.array([jrng.random() < matryoshka for _ in range(n)])[:, None]
        low = (mx.arange(w) < w // 2)[None, :]
        msg = mx.where(cut & ~low, 0.0, msg)
    return tokens, q, f, a, ans, msg


def setup_trainable(core: Core, arm: str, site: str):
    core.freeze()
    if arm in ("c6", "c7"):
        core.backbone.unfreeze()
    if arm in ("c6", "c6r", "c6a", "c6l"):
        core.ports[site].unfreeze()
        core.ports[site].tap.freeze()                   # the mock provider never reads the tap
        if arm == "c6a":
            core.ports[site].reader.freeze()            # unused when the adapter reads (review)
        if arm == "c6l":
            for blk_lora in core.lora.values():
                blk_lora.unfreeze()


def train(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    cfg = CoreConfig(bus_width=args.bus_width, tau=args.tau, layers=args.layers, d=args.d,
                     ffn=4 * args.d, heads=args.heads)
    core = Core(cfg)                                    # identical init for c6/c7 at the same seed
    base_tau = None
    if args.arm == "c6a":
        # C6A (amendment candidate, 2026-11-22): the frozen C7 backbone plus a small nonlinear adapter reader
        # at the site, trained like C6R. Asks whether a cheap adapter, not bus-training, makes a core extensible.
        base, base_ckpt = C.load_core(args.from_core)
        if dataclasses.replace(base.cfg, tau=cfg.tau) != cfg:
            raise SystemExit("c6a: --from-core config differs from requested config (other than tau)")
        base_tau = base.cfg.tau
        acfg = dataclasses.replace(base.cfg, tau=cfg.tau, reader_kind=args.adapter_kind, adapter_hidden=args.adapter_hidden)
        if json.loads((Path(args.from_core) / "receipt.json").read_text()).get("arm") != "c7":
            raise SystemExit("c6a: --from-core must be a c7 core")
        core = Core(acfg)
        core.load_weights(list(tree_flatten(base.parameters())), strict=False)
    elif args.arm == "c6l":
        # C6L (H27): the frozen C7 backbone + a small linear reader + LoRA on the blocks after the site, all
        # applied only when a tile is admitted. Asks whether a loaded tile can pull an untrained core.
        base, base_ckpt = C.load_core(args.from_core)
        if json.loads((Path(args.from_core) / "receipt.json").read_text()).get("arm") != "c7":
            raise SystemExit("c6l: --from-core must be a c7 core")
        if dataclasses.replace(base.cfg, tau=cfg.tau) != cfg:
            raise SystemExit("c6l: --from-core config differs from requested config (other than tau)")
        base_tau = base.cfg.tau
        lcfg = dataclasses.replace(base.cfg, tau=cfg.tau, lora_rank=args.lora_rank, lora_site=args.site)
        core = Core(lcfg)
        core.load_weights(list(tree_flatten(base.parameters())), strict=False)
        port = core.ports[args.site]
        port.reader.weight = port.reader.weight * args.reader_init
    elif args.arm == "c6r":
        if json.loads((Path(args.from_core) / "receipt.json").read_text()).get("arm") != "c7":
            raise SystemExit("c6r: --from-core must be a c7 core")
        base, base_ckpt = C.load_core(args.from_core)
        # tau may differ: c6r only trains the bound-limited port, and a typed token moves the c7 residual
        # by about 1.0 x ||h|| at every site (pilot 2), beyond a 0.5 bound. The receipt records both.
        if dataclasses.replace(base.cfg, tau=cfg.tau) != cfg:
            raise SystemExit("c6r: --from-core config differs from requested config (other than tau)")
        base_tau = base.cfg.tau
        base.cfg = dataclasses.replace(base.cfg, tau=cfg.tau)
        # c7 never trained its ports, so their readers sit at the full init and every write clips at
        # step 0 (gate gradient zero). Pilot 3 debugging: a small reader lets c6r learn at ingress.
        port = base.ports[args.site]
        port.reader.weight = port.reader.weight * args.reader_init
        core = base
    setup_trainable(core, args.arm, args.site)
    M = C.codebook(cfg.bus_width)
    fams = F.split("train")

    def loss_fn(model, tokens, q, f, a, ans, msg):
        bus = None if msg is None else BusCall(args.site, q, f, msg)
        logits = value_logits(model(tokens, bus), a)
        return nn.losses.cross_entropy(logits, ans, reduction="mean")

    sched = optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, args.warmup),
         optim.cosine_decay(args.lr, max(1, args.steps - args.warmup), args.lr * 0.05)], [args.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    grad_fn = nn.value_and_grad(core, loss_fn)

    state = [core.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_bus(tokens, q, f, a, ans, msg):
        loss, g = grad_fn(core, tokens, q, f, a, ans, msg)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(core, g)
        return loss

    @partial(mx.compile, inputs=state, outputs=state)
    def step_typed(tokens, q, f, a, ans):
        loss, g = grad_fn(core, tokens, q, f, a, ans, None)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(core, g)
        return loss

    # answers are value indices 0..63 (value_logits restricts to value tokens)
    data_rng = random.Random(10_000 + args.seed)        # same data order for c6 and c7 at a seed
    jitter_rng = random.Random(30_000 + args.seed)
    log, t0, first_grads = [], time.time(), None
    for it in range(1, args.steps + 1):
        tokens, q, f, a, ans, msg = build_batch(data_rng, jitter_rng, fams, args.batch, args.arm, args.p_ordinary, M,
                                                args.matryoshka)
        if it == 1:
            first_grads = _first_update_report(core, loss_fn, tokens, q, f, a, ans, msg)
        loss = step_typed(tokens, q, f, a, ans) if msg is None else step_bus(tokens, q, f, a, ans, msg)
        mx.eval(state)
        if it % args.log_every == 0 or it == args.steps:
            row = {"step": it, "loss": float(loss.item()), "elapsed_s": round(time.time() - t0, 1),
                   "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3)}
            if it % (args.log_every * 5) == 0 or it == args.steps:
                row.update(quick_eval(core, args.arm, M, cfg, args.site))
                if args.family_eval:
                    row["families"] = family_eval(core, args.arm, M, args.site)
            log.append(row)
            print(row, flush=True)
            if not math.isfinite(row["loss"]):
                raise SystemExit("non-finite loss")
    # the default shape (d 256, 6 layers) keeps the historical name; other shapes are named, so checkpoints
    # of different widths never share a name (2026-11-22)
    shape = "" if (cfg.d, cfg.layers) == (256, 6) else f"-d{cfg.d}-L{cfg.layers}"
    name = f"ens-a0-core-{args.arm}-{args.site}-w{cfg.bus_width}{shape}-s{args.seed}"
    ckpt = C.save_core(core, out, name)
    receipt = {"schema": "ens.a0.core_receipt.v0", "checkpoint": ckpt, "arm": args.arm, "seed": args.seed, "site": args.site,
               "config": core.cfg.to_json(), "args": vars(args), "params": C.n_params(core),
               "trainable_params": C.n_params(core, trainable=True),
               "codebook_sha256": C.codebook_digest(cfg.bus_width),
               "families": F.registry_record(), "first_update_grads": first_grads,
               "from_core": base_ckpt if args.arm in ("c6r", "c6a", "c6l") else None,
               "base_tau": base_tau,       # c6r at a tau unlike its comparison c6 is a different-budget arm
               "steps_per_s": round(args.steps / max(1e-9, time.time() - t0), 2),
               "log": log, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    return receipt


def _first_update_report(core, loss_fn, tokens, q, f, a, ans, msg):
    """Which parameter groups receive nonzero gradient on the first update (spec step 3)."""
    from mlx.utils import tree_flatten
    _, g = nn.value_and_grad(core, loss_fn)(core, tokens, q, f, a, ans, msg)
    rep = {}
    for k, v in tree_flatten(g):
        group = ".".join(k.split(".")[:3])
        n = float(mx.sqrt(mx.sum(v.astype(mx.float32) ** 2)).item())
        rep[group] = max(rep.get(group, 0.0), n)
    return {k: round(v, 6) for k, v in sorted(rep.items())}


def quick_eval(core, arm, M, cfg, site, n=512):
    """Oracle-provider accuracy on fresh train-family rows, and ordinary-suite accuracy."""
    rng = random.Random(99)
    fams = F.split("train")
    rows = make_rows(rng, fams, n)
    res = {}
    for mode in ("ordinary", "typed" if arm == "c7" else "bus"):
        tokens, q, f, a, ans, zs = C.batch(rows, mode)
        bus = BusCall(site, q, f, M[zs]) if mode == "bus" else None
        pred = mx.argmax(value_logits(core(tokens, bus), a), axis=-1)
        res[f"acc_{mode}"] = round(float(mx.mean(pred == ans).item()), 4)
    return res


def family_eval(core, arm, M, site, n_per=200):
    """H29: per training family, ordinary accuracy and native-channel accuracy (bus with clean M[z] for bus arms,
    the typed field for c7), on fresh rows. Logged only with --family-eval, so default receipts are unchanged."""
    out = {}
    for fam in F.split("train"):
        rng = random.Random(hash_seed(fam.fid))
        rows = [F.sample_row(fam, rng) for _ in range(n_per)]
        row = {}
        for mode in ("ordinary", "typed" if arm == "c7" else "bus"):
            tokens, q, f, a, ans, zs = C.batch(rows, mode)
            bus = BusCall(site, q, f, M[zs]) if mode == "bus" else None
            pred = mx.argmax(value_logits(core(tokens, bus), a), axis=-1)
            row["ord" if mode == "ordinary" else "native"] = round(float(mx.mean(pred == ans).item()), 3)
        out[fam.fid] = row
    return out


def hash_seed(text: str) -> int:
    import hashlib
    return int(hashlib.sha256(f"ens.family_eval:{text}".encode()).hexdigest()[:8], 16)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("c6", "c7", "c6r", "c6a", "c6l"), required=True)
    p.add_argument("--lora-rank", type=int, default=8, help="c6l only: LoRA rank on the blocks after the site")
    p.add_argument("--adapter-hidden", type=int, default=256, help="c6a only: adapter hidden width")
    p.add_argument("--adapter-kind", default="mlp", choices=("mlp", "mlp_state"),
                   help="c6a only: mlp reads the message; mlp_state also reads the field row's state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--bus-width", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--p-ordinary", type=float, default=0.25)
    p.add_argument("--family-eval", action="store_true",
                   help="H29: log per-family ordinary and native-channel accuracy with each periodic eval")
    p.add_argument("--matryoshka", type=float, default=0.0,
                   help="H26: fraction of bus rows whose message keeps only the low half of the bus")
    p.add_argument("--site", default=DEFAULT_SITE, choices=("ingress", "early", "middle", "late"),
                   help="A0 bus site; a development-family choice, recorded in the receipt")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--from-core", help="c6r only: a finished c7 checkpoint directory")
    p.add_argument("--reader-init", type=float, default=1.0,
                   help="c6r only: rescale the c7 port reader before training (0.01 lets it learn at ingress)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.arm in ("c6r", "c6a", "c6l") and not args.from_core:
        p.error(f"--arm {args.arm} needs --from-core")
    train(args)


if __name__ == "__main__":
    main()
