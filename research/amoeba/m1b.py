"""Amoeba M1, remedy 1: one context box, then refit the read (plan section 3.2).

The M1 read failed because the host never gathered English context (an untrained
host read the box as well as H0; two donor layers of attention did far better).
Remedy 1 distils that context into the host's mixer layers, with the loss on the
box's input features only:

    loss = weighted MSE( R(host_state_L), z_box )     z_box = standardised donor box pre-activations

Trainable: the Gated DeltaNet mixers and the read adapter R. Frozen: the FFNs and
the lexicon box (E_donor @ P), so box 0 keeps its provenance. Every gradient token
is charged to the amoeba as context-box distillation (plan 9.3), separately from
the H0 curriculum.

Evaluation reuses M1's protocol unchanged: fresh ridge fits from each host layer
on the same 400 held-out donor-world sequences (never used for distillation).
Control: the same budget applied to an untrained host, which tells whether the
English-free curriculum helped. Also recorded: curriculum skills after (forgetting)
and English NLL after (the host now has some English context).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from research.amoeba import evals, transcoders as T  # noqa: E402
from research.amoeba.curriculum import Curriculum  # noqa: E402
from research.amoeba.donor import Donor  # noqa: E402
from research.amoeba.host import Host  # noqa: E402
from research.amoeba.m0 import DATA, atomic, source_rev, tokenizer  # noqa: E402
from research.amoeba.m1 import english_loss, read_dataset, ridge_eval, task_eval  # noqa: E402

READ_SEED, READ_SEQS = 0, 400            # M1's held-out read set (donor_world rows shuffled with seed 0, first 400)


def distill_stream(tok, T=256, seed=1):
    """Distillation text: donor-world rows NOT in the M1 read set, plus general English."""
    rows = [json.loads(l)["text"] for l in (DATA / "donor_world.jsonl").read_text().splitlines()]
    random.Random(READ_SEED).shuffle(rows)
    held, n = set(), 0
    for r in rows:                           # reproduce read_dataset's selection to exclude it
        if len(tok(r, add_special_tokens=False)["input_ids"]) >= T:
            held.add(r); n += 1
        if n == READ_SEQS:
            break
    world = [r for r in rows if r not in held]
    streams = {"donor_world_rest": world}
    for name, k in (("tinystories_train.jsonl", 6000), ("billsum.jsonl", 1500), ("scifact.jsonl", 2000)):
        streams[name] = [json.loads(l)["text"] for l in (DATA / name).read_text().splitlines()[:k]]
    ids = {}
    for name, rws in streams.items():
        buf = []
        for r in rws:
            buf += tok(r, add_special_tokens=False)["input_ids"] + [151643]
        ids[name] = np.array(buf, dtype=np.int32)
    weights = {"donor_world_rest": 0.5, "tinystories_train.jsonl": 0.2, "billsum.jsonl": 0.15, "scifact.jsonl": 0.15}
    rng = random.Random(seed)

    def batch(B):
        out = []
        for name in rng.choices(list(weights), weights=list(weights.values()), k=B):
            s = ids[name]; o = rng.randrange(0, len(s) - T - 1); out.append(s[o:o + T])
        return mx.array(np.stack(out))
    return batch, {k: int(len(v)) for k, v in ids.items()}, len(held)


def box_targets(donor, tcs, box, ids):
    layers = sorted(box)
    _, caps, _ = donor.forward(ids, record=set(layers), stop_after=max(layers))
    zs, fs = [], []
    for l in layers:
        tc = tcs[l]; f = mx.array(box[l].astype(np.int32)); x = caps[l][0]
        zs.append((x / tc.sx) @ tc.W_e[f].T + tc.b_e[f]); fs.append(tc.encode(x)[..., f] > 0)
    return mx.concatenate(zs, -1), mx.concatenate(fs, -1)


def distill(host, donor, tcs, box, batch, steps, B, lr, read_layer, z_mu, z_sd, log):
    k = int(sum(len(v) for v in box.values()))
    host.read_norm = nn.RMSNorm(512)      # the host residual grows with depth; read it normalised
    host.read = nn.Linear(512, k)
    host.freeze()
    for j, m in enumerate(host.layers):
        if j % 2 == 0:
            m.unfreeze()                                  # mixers only
    host.read.unfreeze(); host.read_norm.unfreeze()

    def loss_fn(h, ids, z, fire):
        st = h.states(ids)[read_layer]
        pred = h.read(h.read_norm(st))
        w = 1.0 + 4.0 * fire.astype(mx.float32)          # firing tokens count 5x
        return mx.sum(w * (pred - z) ** 2) / mx.sum(w * mx.ones_like(z))

    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(1e-5, lr, 200), optim.cosine_decay(lr, max(1, steps - 200), lr * 0.1)], [200]),
        weight_decay=0.01)
    lg = nn.value_and_grad(host, loss_fn)
    hist, t0 = [], time.time()
    for s in range(steps):
        ids = batch(B)
        z, fire = box_targets(donor, tcs, box, ids)
        z = (z - z_mu) / z_sd
        loss, g = lg(host, ids, mx.stop_gradient(z), fire)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(host, g)
        mx.eval(host.trainable_parameters(), opt.state, loss)
        if s % 250 == 0 or s == steps - 1:
            row = {"step": s, "sec": round(time.time() - t0, 1), "loss": float(loss)}
            hist.append(row); log(row)
    host.unfreeze()
    return hist


def refit(host, donor, tok, tcs, box, n_seq):
    rseqs, Z, Fz, _ = read_dataset(donor, tok, tcs, box, n_seq, seed=READ_SEED)
    S, Tn = len(rseqs), len(rseqs[0])
    test = np.repeat(np.arange(S), Tn) % 4 == 0; train = ~test
    ids = mx.array(rseqs)
    st = [[np.array(x) for x in host.states(ids[i:i + 8])] for i in range(0, S, 8)]
    Zf, Ff = Z.reshape(-1, Z.shape[-1]), Fz.reshape(-1, Fz.shape[-1])
    fits = {f"layer{j}": ridge_eval(np.concatenate([b[j] for b in st]).reshape(-1, 512), Zf, Ff, train, test)
            for j in range(len(st[0]))}
    # the trained read adapter itself, on the same held-out tokens (standardised targets)
    return fits, (rseqs, Zf, Ff, test)


def adapter_eval(host, read_layer, data, z_mu, z_sd):
    rseqs, Zf, Ff, test = data
    ids = mx.array(rseqs)
    P = np.concatenate([np.array(host.read(host.read_norm(host.states(ids[i:i + 8])[read_layer]))) for i in range(0, len(rseqs), 8)])
    P = P.reshape(-1, P.shape[-1]) * np.array(z_sd) + np.array(z_mu)
    yt, pt = Zf[test], P[test]
    r2 = 1 - ((yt - pt) ** 2).sum(0) / (((yt - yt.mean(0)) ** 2).sum(0) + 1e-12)
    from research.amoeba.m1 import auroc
    au = [a for a in (auroc(pt[:, f], Ff[test][:, f]) for f in range(Zf.shape[1])) if a is not None]
    return {"r2_mean": float(np.mean(r2)), "auroc_mean": float(np.mean(au))}


def window4(args):
    out = Path(args.out); t0 = time.time()
    donor = Donor(); tok = tokenizer(donor)
    tcs = T.load(Path(args.tc))
    bz = np.load(args.box); box = {int(k[1:]): bz[k] for k in bz.files}
    rec = {"schema": "amoeba.m1.window4.context_box.v1", "donor": donor.artifact, "source_revision": source_rev(),
           "h0": args.h0, "box": args.box, "read_layer": args.read_layer, "started": t0}
    batch, stream_sizes, n_held = distill_stream(tok)
    rec["distill_data"] = {"streams_tokens": stream_sizes, "excluded_read_rows": n_held}
    # standardisation of targets from one batch of distillation text
    zc, _ = box_targets(donor, tcs, box, batch(16))
    z_mu = mx.mean(zc.reshape(-1, zc.shape[-1]), 0); z_sd = mx.std(zc.reshape(-1, zc.shape[-1]), 0) + 1e-3
    steps, B = (20, 2) if args.smoke else (args.steps, 16)
    n_read = 24 if args.smoke else READ_SEQS
    cur = Curriculum(T=256, pool=2048, seed=0)
    vocab = mx.array(cur.vocab()); index = mx.zeros((151936,), dtype=mx.int32).at[vocab].add(mx.arange(vocab.size, dtype=mx.int32))
    eng = evals.offtopic_texts(DATA, tok, n_seq=8 if args.smoke else 32, T=256)
    results = {}
    for arm in ("h0", "untrained"):
        mx.random.seed(0)
        host = Host(donor.embed)
        if arm == "h0":
            host.load_weights(list(mx.load(args.h0).items()), strict=True)
        before = {"tasks": task_eval(host, cur, vocab, index), "english_nll": english_loss(host, eng)}
        logf = open(out / f"distill_{arm}.jsonl", "a")
        hist = distill(host, donor, tcs, box, batch, steps, B, args.lr, args.read_layer, z_mu, z_sd,
                       lambda r: (logf.write(json.dumps(r) + "\n"), logf.flush(),
                                  print("distill", arm, r["step"], r["sec"], round(r["loss"], 4), flush=True)))
        fits, data = refit(host, donor, tok, tcs, box, n_read)
        best = max(fits, key=lambda k: fits[k]["r2_mean"])
        results[arm] = {"before": before, "charged_tokens": steps * B * 256, "loss_hist": hist,
                        "refit": fits, "best_layer": best,
                        "adapter": adapter_eval(host, args.read_layer, data, z_mu, z_sd),
                        "after": {"tasks": task_eval(host, cur, vocab, index), "english_nll": english_loss(host, eng)}}
        mx.save_safetensors(str(out / f"host_{arm}_ctxbox.safetensors"), dict(tree_flatten(host.parameters())))
        rec["arms"] = results
        atomic(out / "window4.json", rec)
        r = results[arm]
        print("refit", arm, json.dumps({k: (round(v["r2_mean"], 3), round(v["auroc_mean"], 3)) for k, v in fits.items()}), flush=True)
        print("arm", arm, json.dumps({"best": best, "adapter": r["adapter"],
                                      "tasks_after": {k: round(v["acc"], 3) for k, v in r["after"]["tasks"].items()},
                                      "english_before": r["before"]["english_nll"], "english_after": r["after"]["english_nll"]}), flush=True)
    b = results["h0"]["refit"][results["h0"]["best_layer"]]
    rec["verdict"] = {"read_pass": b["r2_mean"] >= 0.5 and b["auroc_mean"] >= 0.85,
                      "k1_kill": b["r2_mean"] < 0.3 and b["auroc_mean"] < 0.75,
                      "best_r2": b["r2_mean"], "best_auroc": b["auroc_mean"],
                      "curriculum_helps": b["r2_mean"] > results["untrained"]["refit"][results["untrained"]["best_layer"]]["r2_mean"]}
    rec["finished"] = time.time()
    atomic(out / "window4.json", rec)
    print("verdict", json.dumps(rec["verdict"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window4"])
    ap.add_argument("--out", required=True); ap.add_argument("--tc", required=True)
    ap.add_argument("--box", required=True); ap.add_argument("--h0", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--read-layer", type=int, default=7)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    owned = os.environ.get("AMOEBA_OWNED_OUT")
    dirty = out.exists() and any(p.name not in {"desktop-lease.json", "paired-locks.json"}
                                 and not p.name.startswith(("paired-locks.attempt", ".")) for p in out.iterdir())
    if dirty and not (owned and Path(owned).resolve() == out and not (out / "window4.json").exists()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    window4(args)


if __name__ == "__main__":
    main()
