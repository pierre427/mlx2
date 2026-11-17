"""Amoeba 4B track, window 6: where does zero-shot transfer break, and does a trained wiring work?

Window 5 showed the stitched skill delta reaching frozen Qwen3-4B carries no atlas
knowledge (0% on held-out countries). This window separates the two halves and
then runs the plan's M2 trial with a split that makes the box's contribution
unambiguous.

D1 read side   donor base + Delta(R x_host): the specialist's delta fed from the host's
               stitched state instead of its own. Keeps the skill -> the read is fine.
D2 write side  host + M Delta(x_donor): the delta computed from the donor's true input
               (oracle read), written through M. Fails -> the write stitch is the break.
D3 M2 trial    train the write maps (init M) and a per-layer gate on atlas statements for
               60 countries ("train half"), host and core frozen, R fixed; test on the
               other 60 ("test half"). The adapters never see test-half facts, so any
               gain there comes from the donor core.
               Control: the same training with the random core.
D4 LoRA        rank-16 LoRA on the host MLP outputs at the same 13 layers, trained on the
               same train-half statements and token budget. It cannot know test-half
               facts; it bounds what the wiring could learn by itself.
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
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402

from research.amoeba import evals  # noqa: E402
from research.amoeba.donor import Donor  # noqa: E402
from research.amoeba.m0 import DATA, atomic, source_rev, tokenizer  # noqa: E402
from research.amoeba.m2 import (BOX_LAYERS, MLP_KEYS, atlas, fit_stitches, general_stream, layer_map,  # noqa: E402
                                probes, random_core, with_weights)


def mlp_w(i, z, W):
    g = z @ W[f"model.layers.{i}.mlp.gate_proj.weight"].T
    u = z @ W[f"model.layers.{i}.mlp.up_proj.weight"].T
    return (g * mx.sigmoid(g) * u) @ W[f"model.layers.{i}.mlp.down_proj.weight"].T


def split_countries(spec, seed=5):
    names = [it["country"] for it in spec["items"]]
    random.Random(seed).shuffle(names)
    return set(names[:60]), set(names[60:])


def statements(spec, tok, countries, T, seed=0):
    rng = random.Random(seed)
    items = [it for it in spec["items"] if it["country"] in countries]

    def para():
        ids = []
        while len(ids) < T + 1:
            it = rng.choice(items)
            ids += tok(" " + rng.choice(spec["train_templates"]).format(**it), add_special_tokens=False)["input_ids"]
        return ids[:T + 1]
    return para


# ---------------------------------------------------------------- D1 / D2 (need paired runs on identical tokens)
def paired_eval(donor, host, prompts, lmap, st, ft, base, mode, alpha=1.0, bs=32):
    """mode 'd1': donor base + Delta(R x_host) at layer l. mode 'd2': host + alpha M Delta(x_donor)."""
    inv = {lmap[l]: l for l in BOX_LAYERS}
    correct, n = 0, 0
    for batch in evals._batches(prompts, bs):
        ids = mx.array([b[0] for b in batch])
        if mode == "d1":
            _, caph, _ = host.forward(ids, record=set(lmap.values()), stop_after=max(lmap.values()))
            xh = {l: caph[lmap[l]][0].astype(mx.float32) for l in BOX_LAYERS}

            def hook(i, x, y):
                if i not in xh:
                    return None
                z = mx.concatenate([xh[i], mx.ones(xh[i].shape[:-1] + (1,))], -1) @ st[i]["R"]
                return y + mlp_w(i, z, ft) - mlp_w(i, z, base)
            logits, _, _ = donor.forward(ids, mlp_hook=hook)
        else:
            _, capd, _ = donor.forward(ids, record=set(BOX_LAYERS), stop_after=max(BOX_LAYERS))
            xd = {l: capd[l][0] for l in BOX_LAYERS}

            def hook(i, x, y):
                if i not in inv:
                    return None
                l = inv[i]
                d = mlp_w(l, xd[l], ft) - mlp_w(l, xd[l], base)
                return y + (alpha * (d @ st[l]["M"])).astype(y.dtype)
            logits, _, _ = host.forward(ids, mlp_hook=hook)
        cands = list(batch[0][2])
        sub = np.array(logits[:, -1][:, mx.array(cands)].astype(mx.float32))
        for j, b in enumerate(batch):
            correct += int(sub[j].argmax() == cands.index(b[1])); n += 1
    return correct / n


# ---------------------------------------------------------------- D3 / D4 training
def trained_wiring(donor, host, tok, spec, lmap, st, core, base, train_c, steps, B, lr, T=128, log=print, lora=False):
    """Train either the box wiring (write maps W_l init M_l, gate g_l init 1) or a LoRA on host MLP outputs."""
    inv = {lmap[l]: l for l in BOX_LAYERS}
    if lora:
        rng = np.random.default_rng(0)
        P = {f"A{l}": mx.array((rng.standard_normal((host.d, 16)) / np.sqrt(host.d)).astype(np.float32)) for l in BOX_LAYERS}
        P.update({f"B{l}": mx.zeros((16, host.d)) for l in BOX_LAYERS})
    else:
        P = {f"W{l}": mx.array(st[l]["M"]) for l in BOX_LAYERS}
        P.update({f"g{l}": mx.array(0.5) for l in BOX_LAYERS})
    para, gen = statements(spec, tok, train_c, T), general_stream(tok, T, seed=3)

    def make_hook(P):
        def hook(i, x, y):
            if i not in inv:
                return None
            l = inv[i]
            X = x.astype(mx.float32)
            if lora:
                return y + (X @ P[f"A{l}"] @ P[f"B{l}"]).astype(y.dtype)
            z = mx.concatenate([X, mx.ones(X.shape[:-1] + (1,))], -1) @ st[l]["R"]
            d = mlp_w(l, z, core) - mlp_w(l, z, base)
            return y + (P[f"g{l}"] * (d @ P[f"W{l}"])).astype(y.dtype)
        return hook

    def loss_fn(P, ids):
        logits, _, _ = host.forward(ids[:, :-1], mlp_hook=make_hook(P))
        logits = logits.astype(mx.float32)
        lp = logits - mx.logsumexp(logits, -1, keepdims=True)
        return -mx.mean(mx.take_along_axis(lp, ids[:, 1:, None], -1))

    opt = optim.Adam(learning_rate=lr)
    lg = mx.value_and_grad(loss_fn)
    hist, t0 = [], time.time()
    for s in range(steps):
        rows = [para() for _ in range(B // 2)] + [gen() for _ in range(B - B // 2)]
        loss, g = lg(P, mx.array(rows))
        g, _ = optim.clip_grad_norm(g, 1.0)
        P = opt.apply_gradients(g, P)
        mx.eval(P, opt.state, loss)
        if s % 100 == 0 or s == steps - 1:
            hist.append({"step": s, "sec": round(time.time() - t0, 1), "loss": float(loss)}); log(hist[-1])
    n_params = int(sum(v.size for v in P.values()))
    return make_hook(P), hist, n_params


def window6(args):
    out = Path(args.out); t0 = time.time()
    spec = atlas()
    donor = Donor("Qwen/Qwen3-0.6B"); tok = tokenizer(donor)
    host = Donor(args.host, dtype=mx.bfloat16 if args.host_bf16 else mx.float32)
    ft_all = dict(mx.load(args.specialist))
    base = {k: donor.w[k] for k in ft_all}
    ft = {k: v.astype(mx.float32) for k, v in ft_all.items()}
    lmap = layer_map(donor.L, host.L)
    rec = {"schema": "amoeba.m2.window6.v1", "donor": donor.artifact, "host": host.artifact,
           "specialist": args.specialist, "source_revision": source_rev(), "started": t0, "layer_map": lmap}
    smoke = args.smoke
    st = fit_stitches(donor, host, tok, lmap, n_seq=16 if smoke else args.stitch_seqs)
    train_c, test_c = split_countries(spec)
    cap_test = probes(spec, tok, "capital", test_c); cap_train = probes(spec, tok, "capital", train_c)
    cur_test = probes(spec, tok, "currency", test_c)
    if smoke:
        cap_test, cap_train, cur_test = cap_test[:16], cap_train[:16], cur_test[:16]
    off = evals.offtopic_texts(DATA, tok, n_seq=8 if smoke else 48)
    real = evals.topic_prompts(evals.load_topic(), "heldout", tok)
    if smoke:
        real = real[:16]
    # references
    with with_weights(donor, ft):
        rec["specialist_capital_test"] = evals.topic_accuracy(donor, cap_test)["acc"]
    rec["host_capital_test_no_box"] = evals.topic_accuracy(host, cap_test)["acc"]
    # D1, D2
    rec["d1_donor_with_stitched_read"] = paired_eval(donor, host, cap_test, lmap, st, ft, base, "d1")
    rec["d1_donor_with_stitched_read_random_core"] = paired_eval(donor, host, cap_test, lmap, st, random_core(ft, base), base, "d1")
    rec["d2_host_oracle_read"] = {a: paired_eval(donor, host, cap_test, lmap, st, ft, base, "d2", alpha=a) for a in (0.5, 1.0, 2.0)}
    atomic(out / "window6.json", rec)
    print("diag", json.dumps({k: rec[k] for k in ("specialist_capital_test", "host_capital_test_no_box",
                                                   "d1_donor_with_stitched_read", "d1_donor_with_stitched_read_random_core",
                                                   "d2_host_oracle_read")}), flush=True)
    # D3 / D4
    steps, B = (6, 2) if smoke else (args.steps, 8)
    rc = random_core(ft, base)
    arms = {}
    for name, core, lora in (("box", ft, False), ("box_random_core", rc, False), ("lora_r16", None, True)):
        logf = open(out / f"train_{name}.jsonl", "a")
        hook, hist, n_params = trained_wiring(donor, host, tok, spec, lmap, st, core, base, train_c, steps, B, args.lr,
                                              lora=lora, log=lambda r, n=name: (logf.write(json.dumps(r) + "\n"), logf.flush(),
                                                                                  print("train", n, r, flush=True)))
        arms[name] = {"trainable_params": n_params, "charged_tokens": steps * B * 128, "loss_hist": hist,
                      "capital_train_half": evals.topic_accuracy(host, cap_train, hook=hook)["acc"],
                      "capital_test_half": evals.topic_accuracy(host, cap_test, hook=hook)["acc"],
                      "currency_test_half": evals.topic_accuracy(host, cur_test, hook=hook)["acc"],
                      "real_capitals_heldout": evals.topic_accuracy(host, real, hook=hook)["acc"],
                      "offtopic_nll": evals.offtopic_loss(host, off, hook=hook)}
        rec["arms"] = arms
        atomic(out / "window6.json", rec)
        print("arm", name, json.dumps({k: v for k, v in arms[name].items() if k != "loss_hist"}), flush=True)
    rec["host_offtopic_nll"] = evals.offtopic_loss(host, off)
    rec["chance"] = spec["chance"]
    rec["finished"] = time.time()
    atomic(out / "window6.json", rec)
    print("done", json.dumps({n: (a["capital_train_half"], a["capital_test_half"]) for n, a in arms.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window6"])
    ap.add_argument("--out", required=True); ap.add_argument("--specialist", required=True)
    ap.add_argument("--host", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--host-bf16", action="store_true")
    ap.add_argument("--stitch-seqs", type=int, default=192)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    owned = os.environ.get("AMOEBA_OWNED_OUT")
    dirty = out.exists() and any(p.name not in {"desktop-lease.json", "paired-locks.json"}
                                 and not p.name.startswith(("paired-locks.attempt", ".")) for p in out.iterdir())
    if dirty and not (owned and Path(owned).resolve() == out and not (out / "window6.json").exists()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    window6(args)


if __name__ == "__main__":
    main()
