"""H43b post hoc learning: two arms that add the new papers (B) without retraining the phase-1 tile (Pierre,
2026-11-23: "did we or can we inject further post hoc learning to that tile?" ... "would a memory tile have helped here
if it were anchored to key words or topics, with the orthogonal connections at those nodes").

  second   a separate tile trained on B only (physics_tile --train-split B), combined with the phase-1 tile on A by the
           accept handshake: each tile reports its confidence (mean over slots of its top probability) for a mention,
           and the more confident tile answers. Reports how often the handshake picks the tile that trained on the
           paper.
  memory   no training. Each paper is written as an entry with one key per training question template (9: three
           per retrieval task; review 2026-11-23: the state at the title's closing quote is dominated by the template
           before it, so one averaged key mostly matched other papers written with the same template; recall on
           Qwen3-0.6B rose from 0.46-0.56 to 1.00 with per-template keys). Key = the frozen model's state after the
           bus site at the mention, unit RMS, projected off the top-8 ordinary-text principal directions (H41's
           orthogonal keys); value = the paper's true record. A mention retrieves the paper whose best key is nearest
           (cosine). Memory holds A and B (B written post hoc).
           A privileged upper bound, not a learned tile: values are true records read directly and the store is
           append-only (it cannot forget A). With per-template keys, old_A and new_B query with the written templates
           (near-identity lookup); the meaningful memory cells are held_questions_B and cross. memory_recall on
           unseen papers is 0 by construction.

Both evaluate the same cells as physics_tile --train-split runs (physics_tile.evaluate_posthoc).

  python -m research.ens.physics_posthoc --bus <bus dir> --arm second --tile-a <phase-1 dense run> --tile-b <B run> --out <dir>
  python -m research.ens.physics_posthoc --bus <bus dir> --arm memory --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx

from research.ens import common as C
from research.ens import lm_bus as LB
from research.ens import physics_bus as PB
from research.ens import physics_tile as PT

ORTH_K = 8


def setup(args):
    """The world exactly as physics_tile.run builds it (same selection code and seeds)."""
    brec = json.loads((Path(args.bus) / "train_receipt.json").read_text())
    bargs = argparse.Namespace(**brec["args"])
    if not args.no_memory_gate:
        PB.memory_gate(gib=PB.gate_gib(bargs))
    facts, model, cfg, ckpt, tok, voc = PB.load_world(bargs)
    if ckpt != brec["base_checkpoint"] or facts["sha256"] != brec["facts_sha256"]:
        raise SystemExit("base checkpoint or facts differ from the bus receipt")
    s = brec["port"]
    port = PB.RecordPort(model, s["site"], s["rank"], s["tau"])
    port.load_weights(str(Path(args.bus) / "port.safetensors"))
    port.freeze()
    allp = [p for p in facts["papers"] if "'" not in p["title"]]
    prng = random.Random(123 + args.world_seed)
    unseen_ids = set(p["id"] for p in prng.sample(allp, max(1, len(allp) // 20)))
    unseen = [p for p in allp if p["id"] in unseen_ids]
    papers = [p for p in allp if p["id"] not in unseen_ids]
    split = {"A": [q for q in papers if not PT.in_b(q)], "B": [q for q in papers if PT.in_b(q)]}
    return brec, facts, model, cfg, ckpt, tok, voc, port, split, unseen


def make_batch_inputs(model, tok, port):
    def batch_inputs(rows):
        T, W, Y, M, R = PB.stack(tok, [r[2:] for r in rows])
        H = mx.stop_gradient(PB.prefix(model, T, port.site))
        X = PB.unit_rms(PB.site_states(model, T, port.site, M, h=H))
        return T, W, Y, M, mx.stop_gradient(X), H
    return batch_inputs


def load_tile(d: Path, cfg, voc):
    rec = json.loads((d / "receipt.json").read_text())
    n_out = sum(len(voc.ids[s]) for s in PB.SLOTS)
    dense_h, sparse_h = PT.widths(rec.get("width", PT.DENSE_H))     # runs before --width existed are 64 wide
    if rec["arm"] == "dense":
        t = PT.Tile(cfg.hidden_size, dense_h, n_out, sparse=False)
    else:
        t = PT.Tile(cfg.hidden_size, sparse_h, n_out, sparse=True, density1=0.1)
    t.load_weights(str(d / "tile.safetensors"))
    t.freeze()
    return t, rec


def confidence(logits, voc):
    """Mean over slots of the top softmax probability: (B, K)."""
    out, i = 0.0, 0
    for s in PB.SLOTS:
        n = len(voc.ids[s])
        out = out + mx.max(mx.softmax(logits[..., i:i + n].astype(mx.float32), -1), -1)
        i += n
    return out / len(PB.SLOTS)


def record_logits(voc, papers_for_rows):
    """One-hot 'logits' (large margin) for given records: (B, K, 63)."""
    rows = []
    for ps in papers_for_rows:
        per = []
        for p in ps + [ps[0]] * (2 - len(ps)):
            v = []
            for s in PB.SLOTS:
                oh = [-30.0] * len(voc.ids[s])
                oh[voc.values[s].index(p[s])] = 30.0
                v += oh
            per.append(v)
        rows.append(per)
    return mx.array(rows)


def run(args) -> int:
    out = Path(args.out)
    if (out / "receipt.json").exists():
        print("already finished", flush=True)
        return 0
    now = {k: v for k, v in vars(args).items() if k not in ("max_seconds", "chunk_tag", "no_memory_gate")}
    if out.is_dir() and [q.name for q in out.iterdir()] == ["args.json"]:
        (out / "args.json").unlink()                   # a launch that stopped on a check before any cell (review)
    if not out.exists() or not any(out.iterdir()):
        out = C.fresh_out(args.out)
        C.write_json(out / "args.json", now)
    elif not (out / "args.json").exists() or json.loads((out / "args.json").read_text()) != now:
        raise SystemExit("resume: arguments differ (or args.json missing)")   # review: never merge another run's cells
    brec, facts, model, cfg, ckpt, tok, voc, port, split, unseen = setup(args)
    batch_inputs = make_batch_inputs(model, tok, port)
    t0 = time.time()
    deadline = (t0 + args.max_seconds) if args.max_seconds else None
    cells = PB.EvalCells(out, deadline)
    info = {}
    if args.arm == "second":
        ta, reca = load_tile(Path(args.tile_a), cfg, voc)
        tb, recb = load_tile(Path(args.tile_b), cfg, voc)
        # review 2026-11-23: both tiles fresh (no --init: a continued tile B would have seen A), same world, bus,
        # facts and seed (evaluation rows are seeded by --seed, so all arms must share it)
        if reca["arm"] != recb["arm"] or reca.get("width", PT.DENSE_H) != recb.get("width", PT.DENSE_H):
            raise SystemExit("tiles A and B must share arm and width")   # review: never mix them silently
        for r_, sp in ((reca, "A"), (recb, "B")):
            if r_["world"]["train_split"] != sp or r_.get("init") is not None:
                raise SystemExit(f"tile for split {sp} must be a fresh run trained on split {sp} only")
            if r_["world_seed"] != args.world_seed or r_["seed"] != args.seed:
                raise SystemExit("tile world seed / seed differ from --world-seed / --seed")
            if r_["port_checkpoint"] != brec["port_checkpoint"] or r_["facts_sha256"] != brec["facts_sha256"]:
                raise SystemExit("tile was trained on another bus/port or facts file")
        b_ids = {p["id"] for p in split["B"]}

        def tile_fn(X):
            la, lb = ta(X), tb(X)
            pick_b = (confidence(lb, voc) > confidence(la, voc))[..., None]
            return mx.where(pick_b, lb, la)

        def extra(sub, X):
            pick_b = (confidence(tb(X), voc) > confidence(ta(X), voc)).tolist()
            hit = n_m = 0
            for r, pk in zip(sub, pick_b):
                for k, pid in enumerate(r[1]):
                    hit += int(pk[k] == (pid in b_ids))
                    n_m += 1
            return {"routing_correct_per_mention_x_rows": hit / max(1, n_m) * len(sub)}
        info = {"tile_a": reca["tile_checkpoint"], "tile_b": recb["tile_checkpoint"],
                "note": "routing_correct on unseen_papers only means 'picked tile A'; not a routing measure there"}
    elif args.arm == "memory":
        # write: one key per training question template for every trained paper (A then B)
        rng = random.Random(60_000 + args.seed)
        mem_papers = split["A"] + split["B"]
        writes = [(task, {task: [q]}) for task in PB.RETRIEVAL for q in PB.T_TRAIN[task]]
        keys = []
        for i0 in range(0, len(mem_papers), 16):
            ps = mem_papers[i0:i0 + 16]
            per = []
            for task, temps in writes:
                rows = []
                for p in ps:
                    r = PB.make_row(tok, voc, task, [p], rng, temps)
                    if r is None:
                        raise SystemExit(f"write row too long for paper {p['id']}")
                    rows.append((task, [p["id"]]) + r)
                _, _, _, _, X, _ = batch_inputs(rows)
                per.append(X[:, 0].astype(mx.float32))
            kb = mx.stack(per, axis=1)                     # (b, n_writes, d)
            mx.eval(kb)                                    # bounded peak memory on 4B (review)
            keys.append(kb)
        K = mx.concatenate(keys)                           # (N, n_writes, d)
        # ordinary-text directions at the same site (H41 orth keys): top-ORTH_K principal directions of text states
        text = LB.text_windows(tok, "train")
        trng = random.Random(61_000 + args.seed)
        S = []
        for _ in range(8):
            tt = mx.array([text[trng.randrange(len(text))] for _ in range(16)])
            h = PB.prefix(model, tt, port.site).astype(mx.float32)
            S.append(PB.unit_rms(h[:, 1:]).reshape(-1, h.shape[-1]))
            mx.eval(S[-1])
        S = mx.concatenate(S)
        S = S - mx.mean(S, axis=0, keepdims=True)
        _, _, Vt = mx.linalg.svd(S, stream=mx.cpu)
        P = Vt[:ORTH_K]                                    # (k, d)

        def orth(x):
            x = x - (x @ P.T) @ P
            return x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        Ko = orth(K)
        rec_logits = record_logits(voc, [[p] for p in mem_papers])[:, 0]     # (N, 63)

        n_w = Ko.shape[1]
        Kf = Ko.reshape(-1, Ko.shape[-1])                  # (N * n_writes, d)

        def lookup(X):
            q = orth(X.astype(mx.float32))
            sim = (q @ Kf.T).reshape(*q.shape[:-1], -1, n_w)
            return mx.argmax(mx.max(sim, -1), -1)          # (B, K): paper whose best key is nearest

        def tile_fn(X):
            return rec_logits[lookup(X)]

        idx_of = {p["id"]: i for i, p in enumerate(mem_papers)}

        def extra(sub, X):
            got = lookup(X).tolist()
            hit = n_m = 0
            for r, g_ in zip(sub, got):
                for k, pid in enumerate(r[1]):
                    hit += int(idx_of.get(pid, -1) == g_[k])
                    n_m += 1
            return {"memory_recall_per_mention_x_rows": hit / max(1, n_m) * len(sub)}
        info = {"entries": len(mem_papers), "orth_k": ORTH_K, "keys_per_entry": len(writes),
                "note": "privileged upper bound: true records read directly, append-only store; old_A/new_B query "
                        "with the written templates (near-identity lookup), held_questions_B and cross are the "
                        "meaningful cells; memory_recall on unseen papers is 0 by construction"}
    else:
        raise SystemExit(args.arm)
    ev = PT.evaluate_posthoc(tile_fn, model, port, tok, voc, split, unseen, args, batch_inputs, cells, extra)
    receipt = {"schema": "ens.h43b.posthoc_receipt.v1", "arm": args.arm, "seed": args.seed,
               "world_seed": args.world_seed, "base_checkpoint": ckpt, "port_checkpoint": brec["port_checkpoint"],
               "facts_sha256": facts["sha256"], "info": info, "eval": ev, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print("mean", json.dumps(ev["mean"]), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bus", required=True)
    p.add_argument("--arm", choices=("second", "memory"), required=True)
    p.add_argument("--tile-a", default=None)
    p.add_argument("--tile-b", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world-seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=24)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--no-memory-gate", action="store_true")
    p.add_argument("--chunk-tag", default=None)
    p.add_argument("--out", required=True)
    sys.exit(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
