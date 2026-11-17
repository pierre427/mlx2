"""Amoeba 4B track, window 8: hand-wired neurons, no training.

Can new facts be added to frozen Qwen3-4B by constructing neurons in closed form,
without changing any existing weight and without a gradient step? Each fact
(country, relation) gets one added neuron at one layer, applied at every position:

    y_l += beta * relu(k_f . W (x - mu) - tau_f) * s_f * v_f

  x      the layer's MLP input (post-attention RMSNorm), what the host's own MLP reads
  mu, W  mean and inverse square root covariance of x on general English (whitening),
         which turns the key toward directions ordinary text rarely uses ("quiet space")
  k_f    the fact's key: its mean whitened state at the last prompt token, minus the
         mean over all facts of that relation, normalised
  tau_f  threshold between the fact's own scores and the highest score of any other
         fact or any general-text token, so the neuron is off everywhere else
  v_f    the write: either the averaged gradient of the restricted answer log-prob with
         respect to this layer's output at the last token (one backward pass, no
         optimisation), or the final-norm-weighted unembedding direction of the answer
  beta   one global scale, chosen on the construction prompts

Construction uses prefixes of the TRAINING phrasings; the score uses the held-out
probe phrasings, so a fact only counts if the key recognises the country however it
is asked. Arms: layer scan (separability and accuracy), whitened vs raw key, gradient
vs unembedding write, per-relation modularity, off-topic NLL and firing rate.
The qwen MLP has no bias, so the added neurons are ReLU-with-threshold units in a
runtime hook beside the MLP; the host's weights are untouched and deleting a neuron
reverts exactly.
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
import numpy as np  # noqa: E402

from research.amoeba import evals  # noqa: E402
from research.amoeba.donor import Donor  # noqa: E402
from research.amoeba.m0 import DATA, atomic, source_rev, tokenizer  # noqa: E402
from research.amoeba.m2 import atlas, general_stream, probes  # noqa: E402

BUILD = {
    "capital": ["{country}'s capital is", "The capital city of {country} is", "In {country}, the seat of government is",
                "Q: Which city is the capital of {country}? A:", "The government of {country} meets in",
                "Most embassies in {country} are in"],
    "currency": ["The currency of {country} is the", "In {country} people pay in",
                 "Q: What currency does {country} use? A: the", "Prices in {country} are quoted in"],
}
RELS = ("capital", "currency")


def build_prompts(spec, tok):
    rows = []
    for fi, it in enumerate(spec["items"]):
        for rel in RELS:
            key = f"{rel}_token"
            for t in BUILD[rel]:
                rows.append({"ids": tok(t.format(country=it["country"]), add_special_tokens=False)["input_ids"],
                             "fact": (fi, rel), "target": it[key], "cands": spec["candidates"][rel]})
    return rows


def by_len(rows, bs=64):
    g = {}
    for r in rows:
        g.setdefault((len(r["ids"]), r["fact"][1]), []).append(r)
    for L in sorted(g):
        for i in range(0, len(g[L]), bs):
            yield g[L][i:i + bs]


def last_states(host, rows, layers):
    """MLP input at the last token for every construction prompt, per layer (numpy, fp32)."""
    out = {l: [] for l in layers}; order = []
    for b in by_len(rows):
        _, caps, _ = host.forward(mx.array([r["ids"] for r in b]), record=set(layers), stop_after=max(layers))
        for l in layers:
            out[l].append(np.array(caps[l][0][:, -1].astype(mx.float32)))
        order += b
    return {l: np.concatenate(v) for l, v in out.items()}, order


def general_states(host, tok, layers, n_seq, T=256):
    gen = general_stream(tok, T, seed=123)
    out = {l: [] for l in layers}
    for i in range(0, n_seq, 8):
        ids = mx.array([gen()[:T] for _ in range(8)])
        _, caps, _ = host.forward(ids, record=set(layers), stop_after=max(layers))
        for l in layers:
            out[l].append(np.array(caps[l][0].reshape(-1, host.d).astype(mx.float32)))
    return {l: np.concatenate(v) for l, v in out.items()}


def whitener(G, shrink=1e-2):
    mu = G.mean(0)
    C = np.cov((G - mu).T)
    lam, V = np.linalg.eigh(C)
    eps = shrink * lam.mean()
    W = (V / np.sqrt(np.maximum(lam, 0) + eps)) @ V.T
    return mu.astype(np.float32), W.astype(np.float32), lam


def grad_values(host, rows, layer):
    """Mean gradient of the restricted answer log-prob wrt the layer's MLP output at the last token, per fact."""
    acc = {}
    for b in by_len(rows, 32):
        ids = mx.array([r["ids"] for r in b]); B, T = ids.shape
        cands = mx.array(b[0]["cands"])
        tgt = mx.array([b[0]["cands"].index(r["target"]) for r in b])

        def f(delta):
            def hook(i, x, y):
                if i != layer:
                    return None
                return y + mx.concatenate([mx.zeros((B, T - 1, y.shape[-1]), dtype=y.dtype), delta.astype(y.dtype)], 1)
            logits, _, _ = host.forward(ids, mlp_hook=hook)
            sub = logits[:, -1][:, cands].astype(mx.float32)
            lp = sub - mx.logsumexp(sub, -1, keepdims=True)
            return mx.sum(mx.take_along_axis(lp, tgt[:, None], -1))
        g = np.array(mx.grad(f)(mx.zeros((B, 1, host.d))))[:, 0]
        for r, gi in zip(b, g):
            acc.setdefault(r["fact"], []).append(gi)
    return {k: np.mean(v, 0) for k, v in acc.items()}


def unembed_values(host, spec):
    E = np.array(host.head.astype(mx.float32)); gn = np.array(host.w["model.norm.weight"].astype(mx.float32))
    out = {}
    for rel in RELS:
        cands = spec["candidates"][rel]; mean = E[cands].mean(0)
        for fi, it in enumerate(spec["items"]):
            out[(fi, rel)] = gn * (E[it[f"{rel}_token"]] - mean)
    return out


def construct(X, order, G, values, whiten=True, shrink=1e-2):
    """Keys, thresholds and scales for every fact from last-token states X (rows aligned with order)."""
    if whiten:
        mu, W, _ = whitener(G, shrink)
    else:
        mu, W = G.mean(0).astype(np.float32), np.eye(G.shape[1], dtype=np.float32)
    Xt = (X - mu) @ W
    Gt = (G - mu) @ W
    facts = sorted({r["fact"] for r in order})
    idx = {f: [i for i, r in enumerate(order) if r["fact"] == f] for f in facts}
    rel_mean = {rel: Xt[[i for i, r in enumerate(order) if r["fact"][1] == rel]].mean(0) for rel in RELS}
    K = np.stack([Xt[idx[f]].mean(0) - rel_mean[f[1]] for f in facts])
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    S = Xt @ K.T                                  # construction prompts x facts
    SG = Gt @ K.T                                 # general tokens x facts
    tau, scale, sep, margin = [], [], 0, []
    for j, f in enumerate(facts):
        own = S[idx[f], j]
        mask = np.ones(len(order), bool); mask[idx[f]] = False
        other = max(S[mask, j].max(), SG[:, j].max())
        if own.min() > other:
            sep += 1; t = other + 0.5 * (own.min() - other)
        else:
            t = other
        tau.append(t); scale.append(1.0 / max(own.mean() - t, 1e-6)); margin.append(float(own.min() - other))
    V = np.stack([values[f] / (np.linalg.norm(values[f]) + 1e-12) for f in facts])
    return {"facts": facts, "mu": mu, "W": W, "K": K.astype(np.float32), "tau": np.array(tau, np.float32),
            "scale": np.array(scale, np.float32), "V": V.astype(np.float32), "separable": sep,
            "margin_median": float(np.median(margin))}


def loto_accuracy(X, order, G, whiten=True, shrink=1e-2):
    """Leave-one-template-out nearest-centroid accuracy over facts of the same relation (key separability)."""
    if whiten:
        mu, W, _ = whitener(G, shrink)
    else:
        mu, W = G.mean(0), np.eye(G.shape[1])
    Xt = (X - mu) @ W
    tmpl = {}
    for i, r in enumerate(order):
        tmpl.setdefault(r["fact"], []).append(i)
    ok = n = 0
    for rel in RELS:
        fs = [f for f in tmpl if f[1] == rel]
        T = min(len(tmpl[f]) for f in fs)
        for t in range(T):
            C = np.stack([Xt[[i for k, i in enumerate(tmpl[f]) if k != t]].mean(0) for f in fs])
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            Q = np.stack([Xt[tmpl[f][t]] for f in fs])
            ok += int((np.argmax(Q @ C.T, 1) == np.arange(len(fs))).sum()); n += len(fs)
    return ok / n


def neuron_hook(nw, layer, beta, rel=None, fire=None):
    sel = [j for j, f in enumerate(nw["facts"]) if rel is None or f[1] == rel]
    mu, W = mx.array(nw["mu"]), mx.array(nw["W"])
    K, tau = mx.array(nw["K"][sel]), mx.array(nw["tau"][sel])
    s, V = mx.array(nw["scale"][sel]), mx.array(nw["V"][sel])

    def hook(i, x, y):
        if i != layer:
            return None
        H = mx.maximum((x.astype(mx.float32) - mu) @ W @ K.T - tau, 0) * s
        if fire is not None:
            fire[0] += float(mx.sum(mx.max(H, -1) > 0)); fire[1] += H.shape[0] * H.shape[1]
        return y + (beta * (H @ V)).astype(y.dtype)
    return hook


def build_acc(host, rows, hook):
    """Mean construction accuracy over both relations (scored separately: candidate sets differ)."""
    return float(np.mean([evals.topic_accuracy(host, [(r["ids"], r["target"], r["cands"]) for r in rows if r["fact"][1] == rel],
                                               hook=hook)["acc"] for rel in RELS]))


def window8(args):
    out = Path(args.out); t0 = time.time()
    spec = atlas()
    host = Donor(args.host); tok = tokenizer(host)
    smoke = args.smoke
    if smoke:
        spec = dict(spec, items=spec["items"][:12])
    layers = [8, 9] if smoke else args.layers
    rows = build_prompts(spec, tok)
    cap = probes(spec, tok, "capital"); cur = probes(spec, tok, "currency")
    real = evals.topic_prompts(evals.load_topic(), "heldout", tok)
    off = evals.offtopic_texts(DATA, tok, n_seq=8 if smoke else 48)
    if smoke:
        real = real[:16]
    rec = {"schema": "amoeba.m4.window8.v1", "host": host.artifact, "source_revision": source_rev(), "started": t0,
           "skill": "fict.atlas v1", "facts": len(spec["items"]) * 2, "build_templates": BUILD, "layers": layers,
           "chance": spec["chance"]}

    def save():
        atomic(out / "window8.json", rec)

    def full(hook, fire=None):
        r = {"capital": evals.topic_accuracy(host, cap, hook=hook)["acc"],
             "currency": evals.topic_accuracy(host, cur, hook=hook)["acc"],
             "real_capitals_heldout": evals.topic_accuracy(host, real, hook=hook)["acc"]}
        f = [0.0, 0]
        r["offtopic_nll"] = evals.offtopic_loss(host, off, hook=hook if fire is None else fire(f))
        if fire is not None:
            r["offtopic_token_fire_rate"] = f[0] / max(f[1], 1)
        return r

    rec["base"] = full(None); save(); print("base", json.dumps(rec["base"]), flush=True)
    X, order = last_states(host, rows, layers)
    G = general_states(host, tok, layers, n_seq=16 if smoke else args.general_seqs)
    rec["general_tokens"] = int(next(iter(G.values())).shape[0])
    uv = unembed_values(host, spec)
    betas = [1.0, 4.0] if smoke else [0.5, 1, 2, 4, 8, 16, 32]
    scan, built = {}, {}
    for l in layers:
        gv = grad_values(host, rows, l)
        r = {"loto_white": loto_accuracy(X[l], order, G[l], True), "loto_raw": loto_accuracy(X[l], order, G[l], False)}
        for name, white, vals in (("white_grad", True, gv), ("white_unembed", True, uv), ("raw_grad", False, gv)):
            nw = construct(X[l], order, G[l], vals, whiten=white)
            built[(l, name)] = nw
            sw = {b: build_acc(host, rows, neuron_hook(nw, l, b)) for b in betas}
            r[name] = {"separable": nw["separable"], "margin_median": nw["margin_median"], "build_acc_by_beta": sw}
        scan[l] = r; rec["scan"] = scan; save()
        print("layer", l, json.dumps({k: (v if not isinstance(v, dict) else {"sep": v["separable"], "best": max(v["build_acc_by_beta"].values())})
                                      for k, v in r.items()}), flush=True)
    # choose (layer, beta) per arm on construction accuracy only; report probes
    rec["arms"] = {}
    for name in ("white_grad", "white_unembed", "raw_grad"):
        l, b = max(((l, b) for l in layers for b in betas), key=lambda lb: scan[lb[0]][name]["build_acc_by_beta"][lb[1]])
        nw = built[(l, name)]
        r = {"layer": l, "beta": b, "neurons": len(nw["facts"]),
             "params": int(nw["K"].size + nw["V"].size + 2 * len(nw["facts"])),
             **full(neuron_hook(nw, l, b), fire=lambda f, nw=nw, l=l, b=b: neuron_hook(nw, l, b, fire=f))}
        if name == "white_grad":
            r["capital_neurons_only"] = {"capital": evals.topic_accuracy(host, cap, hook=neuron_hook(nw, l, b, rel="capital"))["acc"],
                                         "currency": evals.topic_accuracy(host, cur, hook=neuron_hook(nw, l, b, rel="capital"))["acc"]}
            r["currency_neurons_only"] = {"capital": evals.topic_accuracy(host, cap, hook=neuron_hook(nw, l, b, rel="currency"))["acc"],
                                          "currency": evals.topic_accuracy(host, cur, hook=neuron_hook(nw, l, b, rel="currency"))["acc"]}
            r["probe_acc_every_layer_same_beta"] = {ll: evals.topic_accuracy(host, cap, hook=neuron_hook(built[(ll, name)], ll, b))["acc"]
                                                    for ll in layers}
        rec["arms"][name] = r; save()
        print("arm", name, json.dumps(r), flush=True)
    rec["finished"] = time.time(); save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24, 30])
    ap.add_argument("--general-seqs", type=int, default=96)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    owned = os.environ.get("AMOEBA_OWNED_OUT")
    dirty = out.exists() and any(p.name not in {"desktop-lease.json", "paired-locks.json"}
                                 and not p.name.startswith(("paired-locks.attempt", ".")) for p in out.iterdir())
    if dirty and not (owned and Path(owned).resolve() == out and not (out / "window8.json").exists()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    window8(args)


if __name__ == "__main__":
    main()
