"""H8 routing probe: two admitted contributor modules on one frozen core, and a router choosing the writer.

Two MLP modules, each trained alone on a different dev law, are admitted together on the frozen bus-trained
core. Each held row from either family must get the right module's message. Arms on the mixed held pool:

  oracle        the row's own module (the single-module ceiling)
  address       H8a, L0-style: an explicit module address per row, dispatched through the admission table.
                Equal to oracle by construction: a plumbing check, not evidence (review F5)
  position_only the floor: the majority module per SLOT position (layout only). Different-structure pairs
                are separable by layout alone, so H8b content routing is judged on a same-structure pair
                (route_dev laws) whose two modules split the key space: only the key can tell them apart
  router_sup    H8b, L2-style: a linear router on the core's site state at the SLOT position (after the op
                tokens, which the key-position tap cannot see), trained on module labels
  router_e2e    H8b trained end to end: softmax-mixed messages through the frozen core's answer loss, hard argmax at eval
  average       the mean of both modules' messages (no routing)
  random        a random module per row
  wrong         the other module

  python -m research.ens.route --core <c6 dir> --module-a <dir> --module-b <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from research.ens import abi
from research.ens import common as C
from research.ens import families as F
from research.ens.contrib import check_core_families, core_site, load_module
from research.ens.evaluate import module_messages
from research.ens.model import BusCall, value_logits


def site_hidden(core, tokens, site):
    """The core's residual at the bus site with no bus write (the state a router at that site reads)."""
    bb = core.backbone
    h = bb.embed_tokens(tokens)
    after = {name: i for name, i in core.cfg.sites}[site]
    if after == -1:
        return h
    for i, blk in enumerate(bb.blocks):
        h = blk(h)
        if i == after:
            return h
    raise ValueError(site)


def key_split(fa, fb):
    """For two modules on the same structure, give each half of the key space (content routing by key)."""
    keys = sorted(range(16), key=lambda x: hashlib.sha256(f"{fa.digest}:{fb.digest}:{x}".encode()).hexdigest())
    return set(keys[:8]), set(keys[8:])


def _mixed(fa, fb, split, keys=None):
    ra, rb = F.family_rows(fa, split), F.family_rows(fb, split)
    if keys is not None:
        ra = [r for r in ra if r.x in keys[0]]
        rb = [r for r in rb if r.x in keys[1]]
    rows = [(r, 0) for r in ra] + [(r, 1) for r in rb]
    random.Random(0).shuffle(rows)
    return rows


def _messages(core, site, mods, rows):
    tokens, q, f, a, ans, zs = C.batch([r for r, _ in rows], "bus")
    msgs = mx.stack([module_messages(core, site, m, tokens, q, f, zs) for m in mods], axis=1)   # (B, 2, w)
    lab = mx.array([l for _, l in rows])
    return tokens, q, f, a, ans, msgs, lab


def _read_pos(q, f, where):
    return q if where == "key" else f - 1


def train_router(core, site, mods, rows, *, e2e: bool, where="slot", steps=1500, batch=128, lr=1e-3, seed=0):
    mx.random.seed(seed)
    router = nn.Linear(core.cfg.d, 2)
    rng = random.Random(seed)

    def loss_fn(rt, tokens, q, f, a, ans, msgs, lab):
        h = site_hidden(core, tokens, site)[mx.arange(tokens.shape[0]), _read_pos(q, f, where)]
        logits = rt(h)
        if not e2e:
            return nn.losses.cross_entropy(logits, lab, reduction="mean")
        w = mx.softmax(logits, axis=-1)[..., None]
        msg = mx.sum(w * msgs, axis=1)
        return nn.losses.cross_entropy(value_logits(core(tokens, BusCall(site, q, f, msg)), a), ans, reduction="mean")

    opt = optim.Adam(learning_rate=lr)
    g = nn.value_and_grad(router, loss_fn)
    for _ in range(steps):
        b = [rows[rng.randrange(len(rows))] for _ in range(batch)]
        loss, gr = g(router, *_messages(core, site, mods, b))
        opt.update(router, gr)
        mx.eval(router.parameters(), opt.state)
    return router, float(loss.item())


def probe(core, site, mods, fams, addresses, *, steps=1500, seed=0) -> dict:
    same_layout = (fams[0].cls, fams[0].knobs) == (fams[1].cls, fams[1].knobs)
    keys = key_split(*fams) if same_layout else None
    train_rows = _mixed(*fams, "train", keys)
    held = _mixed(*fams, "held", keys)
    tokens, q, f, a, ans, msgs, lab = _messages(core, site, mods, held)
    B = len(held)
    idx = mx.arange(B)

    def answer(msg):
        return mx.argmax(value_logits(core(tokens, BusCall(site, q, f, msg), check_finite=True), a), -1)

    own, other = answer(msgs[idx, lab]), answer(msgs[idx, 1 - lab])
    disagree = own != other            # rows where routing can matter at all (review F4)

    def acc(msg):
        hit = answer(msg) == ans
        out = {"all": float(mx.mean(hit).item()),
               "disagree_rows": float((mx.sum(hit & disagree) / mx.maximum(mx.sum(disagree), 1)).item())}
        for k in (0, 1):
            out[fams[k].fid] = float((mx.sum(hit & (lab == k)) / mx.sum(lab == k)).item())
        return out

    res = {"n": B, "families": [x.fid for x in fams], "same_layout": same_layout,
           "key_split": [sorted(k) for k in keys] if keys else None,
           "n_disagree": int(mx.sum(disagree).item())}
    res["oracle"] = acc(msgs[idx, lab])
    # H8a: an explicit address header per row, resolved through the admission table
    table = {addr: i for i, addr in enumerate(addresses)}
    header = [addresses[l] for l in lab.tolist()]
    res["address"] = acc(msgs[idx, mx.array([table[h] for h in header])])
    res["wrong"] = acc(msgs[idx, 1 - lab])
    res["average"] = acc(mx.mean(msgs, axis=1))
    rnd = mx.array([random.Random(seed + 7 + i).randrange(2) for i in range(B)])
    res["random"] = acc(msgs[idx, rnd])
    hs = site_hidden(core, tokens, site)
    reads = {"slot": hs[idx, f - 1], "key": hs[idx, q]}
    # floor: a router that sees only the SLOT position (layout), not content (review F1)
    pos_onehot = (mx.arange(F.SEQ_LEN)[None, :] == (f - 1)[:, None]).astype(mx.float32)
    tr_tokens, _, tr_f, *_ = C.batch([r for r, _ in train_rows], "bus")
    tr_lab = mx.array([l for _, l in train_rows])
    tr_pos = (mx.arange(F.SEQ_LEN)[None, :] == (tr_f - 1)[:, None]).astype(mx.float32)
    counts = mx.stack([mx.sum(tr_pos * (tr_lab == k)[:, None], axis=0) for k in (0, 1)], axis=1)   # (T, 2)
    pos_choice = mx.argmax(pos_onehot @ counts + mx.array([1e-3, 0.0]), -1)    # majority label per position
    res["position_only"] = acc(msgs[idx, pos_choice])
    res["position_only"]["route_acc"] = float(mx.mean(pos_choice == lab).item())
    # floor 2: the majority module per key token (review, 2026-11-21). With a split key space this is a
    # perfect lookup by construction, so a key-position router is judged against it: it shows the core keeps
    # key identity there (L0-style lookup), not that routing was inferred from content.
    tr_key = mx.array([r.x for r, _ in train_rows]); key = mx.array([r.x for r, _ in held])
    kc = mx.stack([mx.sum(((tr_key[:, None] == mx.arange(16)[None, :]) & (tr_lab == k)[:, None]).astype(mx.float32), 0)
                   for k in (0, 1)], 1)                                                   # (16, 2)
    key_choice = mx.argmax(kc[key] + mx.array([1e-3, 0.0]), -1)
    res["key_only"] = acc(msgs[idx, key_choice])
    res["key_only"]["route_acc"] = float(mx.mean(key_choice == lab).item())
    # routers read the SLOT position (after the op; the core may have dropped the key there) or the key
    # position itself (H8 follow-up: read where the content is)
    for name, e2e, where in (("router_sup", False, "slot"), ("router_e2e", True, "slot"),
                             ("router_key_sup", False, "key"), ("router_key_e2e", True, "key")):
        rt, last = train_router(core, site, mods, train_rows, e2e=e2e, where=where, steps=steps, seed=seed)
        logits = rt(reads[where])
        choice = mx.argmax(logits, -1)
        res[name] = acc(msgs[idx, choice])
        res[name]["route_acc"] = float(mx.mean(choice == lab).item())
        res[name]["last_train_loss"] = last
        if e2e:   # what it was trained on: the soft mix, and how decided it is (review F3)
            w = mx.softmax(logits, -1)
            res[name]["soft_mix"] = acc(mx.sum(w[..., None] * msgs, axis=1))
            res[name]["mean_max_weight"] = float(mx.mean(mx.max(w, -1)).item())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True)
    ap.add_argument("--module-a", required=True)
    ap.add_argument("--module-b", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    check_core_families(args.core)
    site = core_site(args.core)
    mods, fams, addrs, pkgs = [], [], [], []
    for d in (args.module_a, args.module_b):
        rec = json.loads((Path(d) / "receipt.json").read_text())
        if rec["core_checkpoint"] != ckpt:
            raise SystemExit(f"{d} was built for {rec['core_checkpoint']}, not {ckpt}")
        fam = F.BY_ID[rec["family"]["fid"]]
        if rec["family_digest"] != fam.digest:
            raise SystemExit(f"{d}: trained on a different hidden law")
        if fam.split in F.SEALED:
            raise SystemExit(f"{fam.fid} is sealed")
        if rec.get("kind", "mlp") != "mlp":
            raise SystemExit(f"{d}: only contributor (mlp) modules can be routed; an oracle table reads the true z")
        if rec.get("site") != site:
            raise SystemExit(f"{d}: module site {rec.get('site')} != core site {site}")
        man = json.loads((Path(d) / "manifest.json").read_text())
        mods.append(load_module(d, core.cfg.bus_width)); fams.append(fam)
        addrs.append(man["address"]); pkgs.append(abi.package_digest(man))
    if fams[0].fid == fams[1].fid:
        raise SystemExit("the two modules must serve different families")
    res = probe(core, site, mods, fams, addrs, steps=args.steps)
    receipt = {"schema": "ens.a0.route_receipt.v0", "hypothesis": "H8", "core_checkpoint": ckpt, "site": site,
               "modules": [{"address": a, "package_digest": p} for a, p in zip(addrs, pkgs)],
               "results": res, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
