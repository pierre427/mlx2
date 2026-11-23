"""Train several contributor tiles against one frozen core in a single batched, compiled loop (speed-up).

It is equivalent to running `research.ens.contrib` once per family with the same arguments:
- each tile is built after mx.random.seed(seed), so starting weights are identical;
- each tile has its own data stream random.Random(20000 + seed) over its own pool, so batches are identical;
- the loss is the sum of each tile's own mean cross-entropy, so each tile receives exactly its own gradient;
- the optimiser (AdamW, weight decay 0) is elementwise, so the updates are identical.

Outputs are the same files, in the same format, as contrib.py writes (module.safetensors, manifest.json,
receipt.json) in <out-root>/<prefix><family>, so evaluate.py and decide.py read them unchanged. The receipt
adds "trainer": "contrib_batched" and the list of tiles trained together.

  python -m research.ens.contrib_batched --core <dir> --families a b c --kind mlp --out-root <dir> --prefix mod-c6-mlp-
"""
from __future__ import annotations

import argparse
import random
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import abi
from research.ens import common as C
from research.ens import families as F
from research.ens import vocab as V
from research.ens.contrib import RECIPES, build, check_core_families, core_site, manifest
from research.ens.model import BusCall, value_logits

KINDS = ("mlp", "oracle-table", "key-table")


class Bank(nn.Module):
    def __init__(self, mods):
        super().__init__()
        self.mods = mods


def train(args) -> list[dict]:
    if args.kind not in KINDS:
        raise SystemExit(f"--kind must be one of {KINDS}")
    if len(set(args.families)) != len(args.families):
        raise SystemExit("--families must not repeat a family (outputs would collide)")
    outs = [C.fresh_out(Path(args.out_root) / f"{args.prefix}{fid}") for fid in args.families]
    core, core_ckpt = C.load_core(args.core)
    core.freeze()
    check_core_families(args.core)
    site = core_site(args.core)
    fams = [F.BY_ID[fid] for fid in args.families]
    for fam in fams:
        if fam.split not in F.CONTRIBUTOR_SPLITS:
            raise SystemExit(f"{fam.fid}: contributors train on dev/confirm/gen_* families only")
        if fam.split in F.SEALED and not args.unseal_confirm:
            raise SystemExit(f"{fam.fid} is a sealed confirmation family; pass --unseal-confirm")
    pools = [F.family_rows(fam, "train") for fam in fams]
    mods = []
    for fam in fams:
        mx.random.seed(args.seed)                                    # identical to contrib.py's init
        mods.append(build(args.kind, core.cfg.bus_width, hidden=args.hidden, z_domain=fam.z_domain))
    bank = Bank(mods)
    M, Bt = len(mods), args.batch

    # Pools packed once as arrays, and the frozen core's prefix (embed + blocks up to the site) cached once per
    # pool: each step only gathers rows and runs the bus write plus the blocks after it (speed-up, 2026-11-22).
    packed = []
    for pool in pools:
        tokens, q, f, a, ans, zs = C.batch(pool, "bus")
        hp = mx.concatenate([core.prefix(tokens[i:i + 512], site) for i in range(0, tokens.shape[0], 512)], axis=0)
        mx.eval(hp)
        packed.append((tokens, q, f, a, ans, zs, hp))
    port = core.ports[site]

    def loss_fn(bk, tokens, q, f, a, ans, zs, hp):
        rows = mx.arange(tokens.shape[0])
        if args.kind == "mlp":
            tap = port.tap(hp[rows, q])
            msg = mx.concatenate([bk.mods[i](tap[i * Bt:(i + 1) * Bt]) for i in range(M)], axis=0)
        elif args.kind == "key-table":
            x = tokens[rows, q] - V.K0
            msg = mx.concatenate([bk.mods[i](x[i * Bt:(i + 1) * Bt]) for i in range(M)], axis=0)
        else:
            msg = mx.concatenate([bk.mods[i](zs[i * Bt:(i + 1) * Bt]) for i in range(M)], axis=0)
        logits = value_logits(core.suffix(hp, BusCall(site, q, f, msg)), a)
        ce = nn.losses.cross_entropy(logits, ans, reduction="none").reshape(M, Bt)
        per = mx.mean(ce, axis=1)
        return mx.sum(per), per

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    grad_fn = nn.value_and_grad(bank, loss_fn)
    state = [bank.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(tokens, q, f, a, ans, zs, hp):
        (_, per), g = grad_fn(bank, tokens, q, f, a, ans, zs, hp)
        opt.update(bank, g)
        return per

    rngs = [random.Random(20_000 + args.seed) for _ in fams]
    t0, logs = time.time(), [[] for _ in fams]
    for it in range(1, args.steps + 1):
        parts = []
        for i in range(M):                                 # the same index draws as contrib.py, per tile
            idx = mx.array([rngs[i].randrange(len(pools[i])) for _ in range(Bt)])
            parts.append([x[idx] for x in packed[i]])
        batch_arrays = [mx.concatenate([pt[j] for pt in parts], axis=0) for j in range(7)]
        per = step(*batch_arrays)
        mx.eval(state)
        if it % args.log_every == 0 or it == args.steps:
            for i in range(M):
                logs[i].append({"step": it, "loss": round(float(per[i].item()), 5),
                                "elapsed_s": round(time.time() - t0, 1)})
    receipts = []
    for i, (fam, out) in enumerate(zip(fams, outs)):
        wpath = out / "module.safetensors"
        mx.save_safetensors(str(wpath), dict(tree_flatten(mods[i].parameters())))
        man = manifest(core_ckpt, core.cfg, site, fam, wpath, args.kind)
        C.write_json(out / "manifest.json", man)
        pool = pools[i]
        receipt = {"schema": "ens.a0.contrib_receipt.v0", "recipe": RECIPES[args.kind], "kind": args.kind,
                   "hidden": args.hidden, "snap_codebook": False, "core_checkpoint": core_ckpt, "site": site,
                   "family": fam.record(), "family_digest": fam.digest, "seed": args.seed,
                   "budget": {"updates": args.steps, "batch": args.batch, "lr": args.lr, "train_pool": len(pool),
                              "train_pool_distinct": len({r.signature for r in pool})},
                   "package_digest": abi.package_digest(man), "contract_digest": abi.contract_digest(man),
                   "log": logs[i], "assistance": "none", "trainer": "contrib_batched",
                   "trained_with": [f.fid for f in fams], **C.source_revision(), **C.host_record()}
        C.write_json(out / "receipt.json", receipt)
        receipts.append(receipt)
    return receipts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", required=True)
    p.add_argument("--families", nargs="+", required=True)
    p.add_argument("--kind", default="mlp", choices=KINDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--unseal-confirm", action="store_true")
    p.add_argument("--out-root", required=True)
    p.add_argument("--prefix", default="mod-")
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
