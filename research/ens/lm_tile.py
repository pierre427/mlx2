"""ENS Experiment A step 2: knowledge tiles through the frozen Qwen3-0.6B + bus adapter.

A fixed invented world (N entities x 4 attributes, values drawn once from a seed) is knowledge the base model cannot
have. A tile learns it only through the frozen model + adapter's answer loss, the way A0 contributors learned a
family's hidden law. Two tile kinds:

  address   a table indexed by the fact's id (entity, attribute) given from outside: the oracle-addressed tile
            (H8a-like), an upper bound on what the bus can carry for learned messages
  content   a small MLP reading the model's own state at the answer position (after the bus site, before the
            write): it has to recognise which entity and attribute the prompt asks about (H8b-like)

Both train on the train templates. Evaluation on train templates and on held templates (never seen by the tile or
the adapter), against: the base model (no tile), the adapter with the oracle codebook message (the ceiling), and
the content tile on a different world (wrong-tile control: chance if the tile is not just a prior).

Kill measures, on accuracy among the 16 values of the asked attribute (chance 1/16; review 2026-11-23): content-tile
accuracy on held templates < 0.8 x its train-template accuracy (it keyed on wording), or <= max(base model,
majority-value-per-attribute rate) + 0.10 (no knowledge added). Control: the content tile on a value-permuted world
(same names and attributes, values shuffled within each attribute).

  python -m research.ens.lm_tile --adapter <lm_bus out dir> --out <dir>
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
from research.ens import lm_bus as LB
from research.ens import qwen3 as Q
from research.ens.contrib import unit_rms


def make_world(seed: int, n: int):
    rng = random.Random(90_000 + seed)
    names, seen = [], set()
    while len(names) < n:
        e = LB.entity(rng)
        if e not in seen:
            seen.add(e)
            names.append(e)
    return [(e, a, LB.z_of(a, rng.choice(LB.ATTRS[a]))) for e in names for a in LB.ATTR_NAMES]


class AddressTile(nn.Module):
    kind = "address"

    def __init__(self, n_facts, width):
        super().__init__()
        self.table = nn.Embedding(n_facts, width)

    def __call__(self, fact_ids):
        return unit_rms(self.table(fact_ids))


class ContentTile(nn.Module):
    kind = "content"

    def __init__(self, d, width, hidden=1024):
        super().__init__()
        self.l1 = nn.Linear(d, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, width)

    def __call__(self, h_row):
        x = h_row.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return unit_rms(self.l3(nn.silu(self.l2(nn.silu(self.l1(x))))))


def rows(tok, world, rng, n, templates):
    """[(ids, field, z, fact_id)]"""
    out = []
    for _ in range(n):
        k = rng.randrange(len(world))
        e, a, z = world[k]
        ids = tok.encode(rng.choice(templates).format(e=e, a=a))[:LB.SEQ]
        out.append((ids, len(ids) - 1, z, k))
    return out


def batch(tok, rs):
    return (LB.pad([r[0] for r in rs], tok), mx.array([r[1] for r in rs]), mx.array([r[2] for r in rs]),
            mx.array([r[3] for r in rs]))


def load_adapter(model, d: Path):
    rec = json.loads((d / "receipt.json").read_text())
    s = rec["adapter"]
    ad = LB.BusAdapter(model, s["site"], s["width"], s["rank"], s["tau"])
    ad.load_weights(str(d / "adapter.safetensors"))
    with (d / "adapter.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest()[:12] != rec["adapter_checkpoint"].rpartition("@")[2]:
            raise SystemExit(f"{d}: adapter weights differ from the receipt")
    ad.freeze()
    return ad, rec


def run(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    path = Q.snapshot_dir()
    model, cfg, ckpt = Q.load(path, dtype=mx.bfloat16 if args.dtype == "bf16" else mx.float32)
    model.freeze()
    ad, arec = load_adapter(model, Path(args.adapter))
    if arec["base_checkpoint"] != ckpt:
        raise SystemExit(f"adapter trained on {arec['base_checkpoint']}, not {ckpt}")
    tok = LB.Tok(path)
    M = C.codebook(ad.width)
    vid = mx.array(tok.value_ids)
    world = make_world(args.seed, args.entities)
    tiles = {"address": AddressTile(len(world), ad.width), "content": ContentTile(cfg.hidden_size, ad.width)}
    logs, tile_ckpts = {}, {}
    for kind, tile in tiles.items():
        rng = random.Random(100_000 + args.seed + (0 if kind == "address" else 50))

        def lf(t, tk, f, z, k):
            msg = t(k) if kind == "address" else t
            row = LB.forward(model, tk, (f, msg), ad, at=f).astype(mx.float32)
            return nn.losses.cross_entropy(row, vid[z], reduction="mean")
        g = nn.value_and_grad(tile, lf)
        opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
        log, t0 = [], time.time()
        for it in range(1, args.steps + 1):
            tk, f, z, k = batch(tok, rows(tok, world, rng, args.batch, LB.TEMPLATES_TRAIN))
            loss, gr = g(tile, tk, f, z, k)
            opt.update(tile, gr)
            mx.eval(tile.parameters(), opt.state)
            if it % args.log_every == 0 or it == args.steps:
                log.append({"step": it, "loss": round(float(loss.item()), 5), "elapsed_s": round(time.time() - t0, 1)})
                print(kind, log[-1], flush=True)
        tile.freeze()
        tp = out / f"tile-{kind}.safetensors"
        mx.save_safetensors(str(tp), dict(tree_flatten(tile.parameters())))
        with tp.open("rb") as fh:
            tile_ckpts[kind] = f"ens-expA-tile-{kind}-site{ad.site}-s{args.seed}@{hashlib.file_digest(fh, 'sha256').hexdigest()[:12]}"
        logs[kind] = log

    attr_mask = mx.array([[1.0 if v // 16 == a else 0.0 for v in range(LB.N_VALUES)] for a in range(4)])
    # value-permuted world: same entities and attributes, values shuffled within each attribute (review 3)
    prng = random.Random(120_000 + args.seed)
    permuted = list(world)
    for a in LB.ATTR_NAMES:
        idx = [i for i, (_, aa, _) in enumerate(world) if aa == a]
        vals = [world[i][2] for i in idx]
        prng.shuffle(vals)
        for i, v in zip(idx, vals):
            permuted[i] = (world[i][0], a, v)
    # majority value per attribute in the world: what a prior-only tile scores
    majority = {}
    for a in LB.ATTR_NAMES:
        vs = [z for _, aa, z in world if aa == a]
        majority[a] = max(set(vs), key=vs.count)

    def score(fact_rows, arm):
        tally = {"top1": 0.0, "among_values": 0.0, "among_attr": 0.0}
        for a0 in range(0, len(fact_rows), 32):
            sub = fact_rows[a0:a0 + 32]
            tk, f, z, k = batch(tok, sub)
            if arm == "base":
                row = LB.forward(model, tk, None, at=f)
            else:
                msg = {"oracle": lambda: M[z], "address": lambda: tiles["address"](k),
                       "content": lambda: tiles["content"]}[arm]()
                row = LB.forward(model, tk, (f, msg), ad, at=f)
            row = row.astype(mx.float32)
            vals = row[:, vid]
            tally["top1"] += float(mx.sum(mx.argmax(row, -1) == vid[z]).item())
            tally["among_values"] += float(mx.sum(mx.argmax(vals, -1) == z).item())
            tally["among_attr"] += float(mx.sum(mx.argmax(vals + (attr_mask[z // 16] - 1.0) * 1e9, -1) == z).item())
        return {k2: round(v / len(fact_rows), 4) for k2, v in tally.items()}
    ev = {}
    for name, temps in (("train_templates", LB.TEMPLATES_TRAIN), ("held_templates", LB.TEMPLATES_HELD)):
        rng = random.Random(110_000 + args.seed + len(name))
        fr = rows(tok, world, rng, args.eval_n, temps)
        ev[name] = {arm: score(fr, arm) for arm in ("base", "oracle", "address", "content")}
        # controls: the content tile asked about a value-permuted world (same names, other values); a prior-only
        # tile scores the majority-value rate there and here
        fp = rows(tok, permuted, rng, args.eval_n, temps)
        ev[name]["content_on_permuted_world"] = score(fp, "content")
        ev[name]["majority_value_rate"] = round(sum(r[2] == majority[world[r[3]][1]] for r in fr) / len(fr), 4)
        ev[name]["chance_among_attr"] = round(1 / 16, 4)
    he = ev["held_templates"]["content"]["among_attr"]
    tr = ev["train_templates"]["content"]["among_attr"]
    floor = max(ev["held_templates"]["base"]["among_attr"], ev["held_templates"]["majority_value_rate"])
    kills = {"metric": "among_attr (16-way within the asked attribute)",
             "keyed_on_wording": {"held": he, "train": tr, "killed": he < 0.8 * tr},
             "no_knowledge_added": {"held": he, "floor": floor, "killed": he <= floor + 0.10}}
    kills["any_killed"] = kills["keyed_on_wording"]["killed"] or kills["no_knowledge_added"]["killed"]
    receipt = {"schema": "ens.expA.lm_tile_receipt.v1", "base_checkpoint": ckpt,
               "adapter_checkpoint": arec["adapter_checkpoint"], "adapter": arec["adapter"], "seed": args.seed,
               "world": {"entities": args.entities, "facts": len(world),
                         "sha256": hashlib.sha256(json.dumps(world).encode()).hexdigest()[:16]},
               "budget": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "eval_n": args.eval_n},
               "dtype": args.dtype, "tile_checkpoints": tile_ckpts,
               "params": {k: C.n_params(t) for k, t in tiles.items()}, "logs": logs, "eval": ev, "kill_measures": kills,
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print("eval", json.dumps(ev), "\nkills", json.dumps(kills), flush=True)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--entities", type=int, default=256)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--log-every", type=int, default=100)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
