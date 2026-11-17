"""Amoeba milestone M1: H0 and translation feasibility (plan sections 3.2, 10).

window3:
  1. Pretrain H0 on the English-free curriculum (charged tokens recorded).
  2. S0: H0's English loss vs a unigram model and uniform, plus an untrained host.
  3. Read fit: ridge from host state (each host layer) to the M0 box's transcoder
     pre-activations on donor-world text, held-out R^2 and firing AUROC.
     Controls: untrained host, lexicon only (E_donor P), the donor's own token
     embedding, and an early donor residual (layer 2) as an attention-context reference.
  4. Write fit: how much of the box's writes, read through the donor's final norm
     and unembedding, survives the host's tied 512-d lexicon (logit-lens R^2).

Pass (plan M1): read R^2 >= 0.5 and firing AUROC >= 0.85 on held-out tokens.
Kill K1: R^2 < 0.3 and AUROC < 0.75 after one context-box attempt (not run here).
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
from mlx.utils import tree_flatten  # noqa: E402

from research.amoeba import evals, transcoders as T  # noqa: E402
from research.amoeba.curriculum import TASKS, Curriculum  # noqa: E402
from research.amoeba.donor import Donor  # noqa: E402
from research.amoeba.host import Host, n_params  # noqa: E402
from research.amoeba.m0 import DATA, atomic, source_rev, tokenizer  # noqa: E402


def masked_loss(host, x, m, vocab, index):
    logits = host(x[:, :-1], vocab=vocab)
    tgt = index[x[:, 1:]]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0]
    w = m[:, 1:]
    return mx.sum(nll * w) / mx.maximum(mx.sum(w), 1.0), logits


def task_eval(host, cur, vocab, index, n=64):
    rng_state = cur.rng.bit_generator.state
    cur.rng = np.random.default_rng(12345)
    out = {}
    for task in TASKS:
        x, m, _ = cur.batch(n, tasks=(task,))
        x, m = mx.array(x), mx.array(m)
        loss, logits = masked_loss(host, x, m, vocab, index)
        pred = mx.argmax(logits, axis=-1)
        ok = (pred == index[x[:, 1:]]) * m[:, 1:]
        out[task] = {"loss": float(loss), "acc": float(mx.sum(ok) / mx.sum(m[:, 1:]))}
    cur.rng = np.random.default_rng(); cur.rng.bit_generator.state = rng_state
    return out


def pretrain(host, cur, steps, B, lr, log, ckpt=None):
    vocab = mx.array(cur.vocab())
    index = mx.zeros((151936,), dtype=mx.int32).at[vocab].add(mx.arange(vocab.size, dtype=mx.int32))
    warm = min(500, max(1, steps // 10))
    sched = optim.join_schedules([optim.linear_schedule(1e-5, lr, warm), optim.cosine_decay(lr, max(1, steps - warm), lr * 0.05)], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    lg = nn.value_and_grad(host, lambda mdl, x, m: masked_loss(mdl, x, m, vocab, index)[0])
    hist, t0 = [], time.time()
    for s in range(steps):
        x, m, _ = cur.batch(B)
        loss, g = lg(host, mx.array(x), mx.array(m))
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(host, g)
        mx.eval(host.parameters(), opt.state, loss)
        if s % 500 == 0 or s == steps - 1:
            if ckpt is not None and s > 0:
                mx.save_safetensors(str(ckpt), dict(tree_flatten(host.parameters())))
            row = {"step": s, "sec": round(time.time() - t0, 1), "loss": float(loss),
                   "tasks": task_eval(host, cur, vocab, index)}
            hist.append(row); log(row)
    return hist, vocab, index


def english_loss(host_or_none, seqs, *, unigram=None):
    tot, n = 0.0, 0
    for i in range(0, len(seqs), 8):
        b = np.array(seqs[i:i + 8])
        if host_or_none is None:
            nll = -np.log(unigram[b[:, 1:]]); tot += float(nll.sum()); n += nll.size; continue
        logits = host_or_none(mx.array(b[:, :-1]))
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(lp, mx.array(b[:, 1:])[..., None], axis=-1)[..., 0]
        tot += float(mx.sum(nll)); n += nll.size
    return tot / n


def unigram_model(tok, V, max_rows=20000):
    counts = np.ones(V)                         # add-one smoothing
    rows = [json.loads(l)["text"] for l in (DATA / "tinystories_train.jsonl").read_text().splitlines()[:max_rows]]
    for r in rows:
        np.add.at(counts, tok(r, add_special_tokens=False)["input_ids"], 1)
    return counts / counts.sum()


def read_dataset(donor, tok, tcs, box, n_seq, T=256, seed=0):
    """Donor box pre-activations and firing labels on donor-world text (+ the topic prompts' tails)."""
    rows = [json.loads(l)["text"] for l in (DATA / "donor_world.jsonl").read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    seqs = []
    for r in rows:
        ids = tok(r, add_special_tokens=False)["input_ids"]
        if len(ids) >= T:
            seqs.append(ids[:T])
        if len(seqs) == n_seq:
            break
    layers = sorted(box)
    Z, Fz, X2 = [], [], []
    for i in range(0, len(seqs), 8):
        b = mx.array(seqs[i:i + 8])
        _, caps, _ = donor.forward(b, record=set(layers) | {2}, stop_after=max(layers))
        z_parts, f_parts = [], []
        for l in layers:
            tc = tcs[l]; ids = mx.array(box[l].astype(np.int32))
            x = caps[l][0]
            pre = (x / tc.sx) @ tc.W_e[ids].T + tc.b_e[ids]
            fire = tc.encode(x)[..., ids] > 0
            z_parts.append(pre); f_parts.append(fire)
        Z.append(np.array(mx.concatenate(z_parts, -1))); Fz.append(np.array(mx.concatenate(f_parts, -1)))
        X2.append(np.array(caps[2][0]))
    return seqs, np.concatenate(Z), np.concatenate(Fz), np.concatenate(X2)


def auroc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) < 10 or len(neg) < 10:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ridge_eval(Xf, Z, Fz, train, test, lam=1.0):
    Xm, Xs = Xf[train].mean(0), Xf[train].std(0) + 1e-6
    A = np.c_[(Xf - Xm) / Xs, np.ones(len(Xf))].astype(np.float64)
    W = np.linalg.solve(A[train].T @ A[train] + lam * np.eye(A.shape[1]), A[train].T @ Z[train])
    P = A[test] @ W
    yt = Z[test]
    r2_feat = 1 - ((yt - P) ** 2).sum(0) / (((yt - yt.mean(0)) ** 2).sum(0) + 1e-12)
    au = [auroc(P[:, f], Fz[test][:, f]) for f in range(Z.shape[1])]
    au = [a for a in au if a is not None]
    pos = Fz[test].any(0)
    return {"r2_mean": float(np.mean(r2_feat)), "r2_median": float(np.median(r2_feat)),
            "r2_variance_weighted": float(1 - ((yt - P) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum()),
            "auroc_mean": float(np.mean(au)) if au else None, "auroc_features": len(au),
            "features_with_positives": int(pos.sum())}


def window3(args):
    out = Path(args.out)
    t0 = time.time()
    donor = Donor(); tok = tokenizer(donor)
    tcs = T.load(Path(args.tc))
    bz = np.load(args.box); box = {int(k[1:]): bz[k] for k in bz.files}
    rec = {"schema": "amoeba.m1.window3.v1", "donor": donor.artifact, "source_revision": source_rev(),
           "box": args.box, "box_size": int(sum(len(v) for v in box.values())), "started": t0}
    E = donor.embed
    mx.random.seed(0)
    host = Host(E)
    untrained = Host(E)
    rec["host_trainable_params"] = n_params(host)
    cur = Curriculum(T=256, pool=args.pool, seed=0)
    logf = open(out / "h0_train_log.jsonl", "a")
    steps, B = (30, 4) if args.smoke else (args.steps, 32)
    hist, vocab, index = pretrain(host, cur, steps, B, args.lr,
                                  lambda r: (logf.write(json.dumps(r) + "\n"), logf.flush(),
                                             print("h0", r["step"], r["sec"], round(r["loss"], 3),
                                                   {k: round(v["acc"], 3) for k, v in r["tasks"].items()}, flush=True)),
                                  ckpt=out / "h0.ckpt.safetensors")
    rec["h0"] = {"steps": steps, "batch": B, "seq": 256, "pool": args.pool, "lr": args.lr, "charged_tokens": steps * B * 256,
                 "final_tasks": hist[-1]["tasks"], "untrained_tasks": task_eval(untrained, cur, vocab, index)}
    mx.save_safetensors(str(out / "h0.safetensors"), dict(tree_flatten(host.parameters())))
    atomic(out / "window3.json", rec)
    # S0
    seqs = evals.offtopic_texts(DATA, tok, n_seq=32 if not args.smoke else 4, T=256)
    uni = unigram_model(tok, E.shape[0], max_rows=20000 if not args.smoke else 500)
    rec["s0"] = {"h0_english_nll": english_loss(host, seqs), "untrained_english_nll": english_loss(untrained, seqs),
                 "unigram_nll": english_loss(None, seqs, unigram=uni), "uniform_nll": float(np.log(E.shape[0])),
                 "donor_english_nll": evals.offtopic_loss(donor, seqs)}
    rec["s0"]["pass_no_english_beyond_unigram"] = rec["s0"]["h0_english_nll"] >= rec["s0"]["unigram_nll"] * 0.97
    atomic(out / "window3.json", rec)
    print("s0", json.dumps(rec["s0"]), flush=True)
    # read fit
    n_seq = 24 if args.smoke else args.read_seqs
    rseqs, Z, Fz, X2 = read_dataset(donor, tok, tcs, box, n_seq)
    S = len(rseqs); Tn = len(rseqs[0])
    seq_id = np.repeat(np.arange(S), Tn)
    test = seq_id % 4 == 0; train = ~test
    feats = {}
    ids_all = mx.array(rseqs)
    def host_states(h):
        st = []
        for i in range(0, S, 8):
            st.append([np.array(s) for s in h.states(ids_all[i:i + 8])])
        return [np.concatenate([b[j] for b in st]).reshape(-1, 512) for j in range(len(st[0]))]
    for j, Xh in enumerate(host_states(host)):
        feats[f"h0_layer{j}"] = Xh
    for j, Xh in enumerate(host_states(untrained)):
        if j in (0, 3, 7):
            feats[f"untrained_layer{j}"] = Xh
    feats["lexicon_only"] = np.concatenate([np.array(host.embed(ids_all[i:i + 8])) for i in range(0, S, 8)]).reshape(-1, 512)
    feats["donor_token_embedding"] = np.concatenate([np.array(E[ids_all[i:i + 8]]) for i in range(0, S, 8)]).reshape(-1, 1024)
    feats["donor_residual_layer2_mlp_in"] = X2.reshape(-1, 1024)
    # context features: host state concatenated with lexicon of the previous 3 tokens (a cheap context box proxy)
    Zf, Ff = Z.reshape(-1, Z.shape[-1]), Fz.reshape(-1, Fz.shape[-1])
    reads = {name: ridge_eval(Xf, Zf, Ff, train, test) for name, Xf in feats.items()}
    best = max((k for k in reads if k.startswith("h0_")), key=lambda k: reads[k]["r2_variance_weighted"])
    rec["read"] = {"tokens": int(len(Zf)), "train_tokens": int(train.sum()), "test_tokens": int(test.sum()),
                   "fits": reads, "best_host_layer": best,
                   "pass": reads[best]["r2_mean"] >= 0.5 and (reads[best]["auroc_mean"] or 0) >= 0.85,
                   "k1_kill_zone": reads[best]["r2_mean"] < 0.3 and (reads[best]["auroc_mean"] or 1) < 0.75}
    atomic(out / "window3.json", rec)
    print("read", json.dumps({k: (round(v["r2_mean"], 3), round(v["r2_variance_weighted"], 3),
                                  None if v["auroc_mean"] is None else round(v["auroc_mean"], 3)) for k, v in reads.items()}), flush=True)
    # write fit (logit-lens): box feature writes u_f = sy W_d[:, f], read by the donor's final norm weight, mapped into
    # the host's tied lexicon space; R^2 measured in donor-logit space through E (Gram of E).
    wn = donor.w["model.norm.weight"]
    U = mx.concatenate([tcs[l].feature_write(mx.array(box[l].astype(np.int32))) for l in sorted(box)], axis=1) * wn[:, None]
    Pt = host.P.T                                               # (512, 1024): host -> donor embedding space
    G = (E.T @ E) / E.shape[0]                                  # (1024,1024) metric of the unembedding
    # best host vector per write: minimise (Pt^T w - u)^T G (Pt^T w - u)
    A = Pt @ G @ Pt.T
    Wb = mx.linalg.solve(A + 1e-6 * mx.eye(A.shape[0]), Pt @ G @ U, stream=mx.cpu)
    R = Pt.T @ Wb - U
    num = mx.sum(R * (G @ R), axis=0); den = mx.sum(U * (G @ U), axis=0)
    r2w = 1 - num / den
    rec["write"] = {"logit_lens_r2_mean": float(mx.mean(r2w)), "logit_lens_r2_min": float(mx.min(r2w)),
                    "note": "fraction of each box write's donor-logit effect expressible through the host's 512-d lexicon"}
    rec["finished"] = time.time()
    atomic(out / "window3.json", rec)
    print("write", json.dumps(rec["write"]), flush=True)
    print("verdict", json.dumps({"read_pass": rec["read"]["pass"], "k1_kill_zone": rec["read"]["k1_kill_zone"],
                                 "best": best, "s0": rec["s0"]["pass_no_english_beyond_unigram"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window3"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tc", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--pool", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--read-seqs", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    owned = os.environ.get("AMOEBA_OWNED_OUT")
    benign = {"desktop-lease.json", "paired-locks.json"}
    dirty = out.exists() and any(p.name not in benign and not p.name.startswith(("paired-locks.attempt", "."))
                                 for p in out.iterdir())
    if dirty and not (owned and Path(owned).resolve() == out and not (out / "window3.json").exists()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    window3(args)


if __name__ == "__main__":
    main()
