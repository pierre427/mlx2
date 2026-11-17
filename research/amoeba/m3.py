"""Amoeba 4B track, window 7: the same-coordinates test.

Windows 5-6 failed to move a skill from Qwen3-0.6B into Qwen3-4B, but they could not
say whether the failure was the translation between models or the box idea itself.
This window removes translation: the donor is a fine-tuned copy of the host, so the
box lives in the host's own coordinates and needs no read or write map.

  J  joint specialist: fine-tune the host's MLPs at 13 layers on fict.atlas (both
     relations) with general-text replay. Box_J(l, z) = MLP_J,l(z) - MLP_base,l(z).
  A  capital-only specialist, B currency-only specialist (same recipe, one relation).

Arms, all on the frozen base host with boxes added to MLP outputs at runtime:
  E1 exactness   Box_J at every layer must reproduce J (max logit diff).
  E2 compression Box_J with each weight delta truncated to rank r (SVD).
  E3 locality    Box_J with one layer dropped, and with one layer alone.
  E4 relocation  Box_J moved s layers up or down: is the box tied to its depth?
  E5 composition A alone, B alone, A+B as two boxes (output sum), A+B weight merge,
                 and the joint J, on both relations: do commits stack without
                 interference, and does removing one leave the other?
  E6 controls    random core (same per-matrix delta norm), off-topic NLL, real-world
                 capitals the host already knew.
Probes use held-out phrasings (PROBE_TEMPLATES never appear in training).
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
from research.amoeba.m2 import MLP_KEYS, atlas, general_stream, probes, random_core, with_weights  # noqa: E402
from research.amoeba.m2b import mlp_w  # noqa: E402

# host layers matching the depth fraction of the 0.6B specialist's layers 6-18 (window 5 layer map)
HOST_LAYERS = [8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23]


def stream(spec, tok, T, relation=None, seed=0):
    rng = random.Random(seed)
    temps = [t for t in spec["train_templates"] if relation is None or "{" + relation + "}" in t]

    def para():
        ids = []
        while len(ids) < T + 1:
            it = rng.choice(spec["items"])
            ids += tok(" " + rng.choice(temps).format(**it), add_special_tokens=False)["input_ids"]
        return ids[:T + 1]
    return para


def finetune(host, spec, tok, layers, steps, B, lr, relation=None, T=128, log=print, check=None):
    keys = [f"model.layers.{l}.{k}" for l in layers for k in MLP_KEYS]
    base = {k: host.w[k] for k in keys}
    params = {k: mx.array(v) for k, v in base.items()}
    para, gen = stream(spec, tok, T, relation), general_stream(tok, T, seed=11)
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(p, ids):
        saved = {k: host.w[k] for k in p}
        host.w.update(p)
        logits, _, _ = host.forward(ids[:, :-1])
        host.w.update(saved)
        lp = logits - mx.logsumexp(logits, -1, keepdims=True)
        return -mx.mean(mx.take_along_axis(lp, ids[:, 1:, None], -1))

    lg = mx.value_and_grad(loss_fn)
    t0 = time.time(); hist = []
    for s in range(steps):
        rows = [para() for _ in range(B // 2)] + [gen() for _ in range(B - B // 2)]
        loss, g = lg(params, mx.array(rows))
        g, _ = optim.clip_grad_norm(g, 1.0)
        params = opt.apply_gradients(g, params)
        mx.eval(params, opt.state, loss)
        if s % 100 == 0 or s == steps - 1:
            r = {"step": s, "sec": round(time.time() - t0, 1), "loss": float(loss)}
            if check is not None and (s % 250 == 0 or s == steps - 1):
                with with_weights(host, params):
                    r["probe_subset_acc"] = check()
            hist.append(r); log(r)
    rel = {k: float(mx.linalg.norm(params[k] - base[k]) / mx.linalg.norm(base[k])) for k in keys}
    return params, base, hist, float(np.mean(list(rel.values())))


def box_hook(boxes, base, alpha=1.0):
    """boxes: list of (weights, {target_layer: source_layer}). Adds alpha * sum Box(src, x) at each target."""
    def hook(i, x, y):
        add = None
        for W, place in boxes:
            if i in place:
                l = place[i]
                d = mlp_w(l, x, W) - mlp_w(l, x, base)
                add = d if add is None else add + d
        return None if add is None else y + (alpha * add).astype(y.dtype)
    return hook


def ident(layers):
    return {l: l for l in layers}


def low_rank(ft, base, ranks):
    """Per matrix, truncated SVD of the weight delta; returns {r: weights} with base + U_r S_r V_r^T."""
    fac = {}
    for k in ft:
        d = np.asarray((ft[k] - base[k]).astype(mx.float32))
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        fac[k] = (U, S, Vt, float((S ** 2).sum()))
    out, energy = {}, {}
    for r in ranks:
        out[r] = {k: base[k] + mx.array((U[:, :r] * S[:r]) @ Vt[:r]) for k, (U, S, Vt, _) in fac.items()}
        energy[r] = float(np.mean([(S[:r] ** 2).sum() / tot for (U, S, Vt, tot) in fac.values()]))
    params = {r: int(sum(r * sum(base[k].shape) for k in ft)) for r in ranks}
    return out, energy, params


def window7(args):
    out = Path(args.out); t0 = time.time()
    spec = atlas()
    host = Donor(args.host)
    tok = tokenizer(host)
    smoke = args.smoke
    layers = HOST_LAYERS[:2] if smoke else HOST_LAYERS
    cap = probes(spec, tok, "capital"); cur = probes(spec, tok, "currency")
    real = evals.topic_prompts(evals.load_topic(), "heldout", tok)
    off = evals.offtopic_texts(DATA, tok, n_seq=8 if smoke else 48)
    if smoke:
        cap, cur, real = cap[:32], cur[:32], real[:16]
    sub = random.Random(1).sample(cap, min(96, len(cap)))
    rec = {"schema": "amoeba.m3.window7.v1", "host": host.artifact, "host_quantization": host.quantization,
           "source_revision": source_rev(), "skill": "fict.atlas v1", "box_layers": layers, "started": t0,
           "chance": spec["chance"], "probe_counts": {"capital": len(cap), "currency": len(cur)}}

    def score(hook=None, offtopic=True, realcaps=True):
        r = {"capital": evals.topic_accuracy(host, cap, hook=hook)["acc"],
             "currency": evals.topic_accuracy(host, cur, hook=hook)["acc"]}
        if realcaps:
            r["real_capitals_heldout"] = evals.topic_accuracy(host, real, hook=hook)["acc"]
        if offtopic:
            r["offtopic_nll"] = evals.offtopic_loss(host, off, hook=hook)
        return r

    def save():
        atomic(out / "window7.json", rec)

    rec["base_host"] = score(); save()
    print("base", json.dumps(rec["base_host"]), flush=True)

    steps = {"J": 6, "A": 4, "B": 4} if smoke else {"J": args.steps_joint, "A": args.steps_single, "B": args.steps_single}
    Bsz = 2 if smoke else 16
    specs, base = {}, None
    rec["specialists"] = {}
    for name, rel in (("J", None), ("A", "capital"), ("B", "currency")):
        logf = open(out / f"specialist_{name}.jsonl", "a")
        W, base, hist, dn = finetune(
            host, spec, tok, layers, steps[name], Bsz, args.lr, relation=rel,
            check=lambda: evals.topic_accuracy(host, sub)["acc"],
            log=lambda r, n=name: (logf.write(json.dumps(r) + "\n"), logf.flush(), print("ft", n, r, flush=True)))
        specs[name] = W
        with with_weights(host, W):
            s = score()
        s.update({"relation": rel or "both", "steps": steps[name], "train_tokens": steps[name] * Bsz * 128,
                  "relative_delta_norm_mean": dn, "loss_hist": hist})
        rec["specialists"][name] = s; save()
        print("specialist", name, json.dumps({k: v for k, v in s.items() if k != "loss_hist"}), flush=True)
        if name == "J" and not smoke:
            mx.save_safetensors(str(out.parent / "specialist_J_bf16.safetensors"),
                                {k: v.astype(mx.bfloat16) for k, v in W.items()})
    J = specs["J"]
    L = ident(layers)

    # E1 exactness
    ids = mx.array([cap[0][0], cap[1][0]] if len(cap[0][0]) == len(cap[1][0]) else [cap[0][0]])
    with with_weights(host, J):
        lj, _, _ = host.forward(ids)
    lb, _, _ = host.forward(ids, mlp_hook=box_hook([(J, L)], base))
    l0, _, _ = host.forward(ids); la0, _, _ = host.forward(ids, mlp_hook=box_hook([(J, L)], base, alpha=0.0))
    rec["E1"] = {"box_vs_specialist_max_abs_logit_diff": float(mx.max(mx.abs(lj - lb))),
                 "alpha0_vs_base_max_abs_logit_diff": float(mx.max(mx.abs(l0 - la0))),
                 "box_J": score(box_hook([(J, L)], base), offtopic=False, realcaps=False)}
    save(); print("E1", json.dumps(rec["E1"]), flush=True)

    # E2 compression
    ranks = [1, 4] if smoke else [1, 4, 16, 64, 256]
    lr_w, energy, nparam = low_rank(J, base, ranks)
    rec["E2"] = {"full_box_params": int(sum(v.size for v in J.values())), "ranks": {}}
    for r in ranks:
        rec["E2"]["ranks"][r] = {"params": nparam[r], "energy_mean": energy[r],
                                 **score(box_hook([(lr_w[r], L)], base), realcaps=r >= 64)}
        save(); print("E2", r, json.dumps(rec["E2"]["ranks"][r]), flush=True)
    del lr_w

    # E3 locality (capital only, to keep it short)
    rec["E3"] = {"drop_one": {}, "only_one": {}}
    for l in layers:
        rec["E3"]["drop_one"][l] = evals.topic_accuracy(host, cap, hook=box_hook([(J, {m: m for m in layers if m != l})], base))["acc"]
        rec["E3"]["only_one"][l] = evals.topic_accuracy(host, cap, hook=box_hook([(J, {l: l})], base))["acc"]
    half = len(layers) // 2
    rec["E3"]["first_half"] = score(box_hook([(J, ident(layers[:half]))], base), offtopic=False, realcaps=False)
    rec["E3"]["second_half"] = score(box_hook([(J, ident(layers[half:]))], base), offtopic=False, realcaps=False)
    save(); print("E3", json.dumps(rec["E3"]), flush=True)

    # E4 relocation: box from layer l applied at layer l+s
    rec["E4"] = {}
    for s in ([1] if smoke else [-2, -1, 1, 2]):
        place = {l + s: l for l in layers if 0 <= l + s < host.L}
        rec["E4"][s] = score(box_hook([(J, place)], base), realcaps=False)
        save(); print("E4", s, json.dumps(rec["E4"][s]), flush=True)

    # E5 composition
    A, Bw = specs["A"], specs["B"]
    merged = {k: A[k] + Bw[k] - base[k] for k in A}
    rec["E5"] = {"A_box": score(box_hook([(A, L)], base)), "B_box": score(box_hook([(Bw, L)], base)),
                 "A_plus_B_boxes": score(box_hook([(A, L), (Bw, L)], base))}
    with with_weights(host, merged):
        rec["E5"]["A_plus_B_weight_merge"] = score()
    rec["E5"]["J_plus_A_boxes"] = score(box_hook([(J, L), (A, L)], base), offtopic=False, realcaps=False)
    save(); print("E5", json.dumps(rec["E5"]), flush=True)

    # E6 random core
    rc = random_core(J, base)
    rec["E6"] = {"random_core_J": score(box_hook([(rc, L)], base))}
    rec["finished"] = time.time(); save()
    print("E6", json.dumps(rec["E6"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window7"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--steps-joint", type=int, default=1000); ap.add_argument("--steps-single", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    owned = os.environ.get("AMOEBA_OWNED_OUT")
    dirty = out.exists() and any(p.name not in {"desktop-lease.json", "paired-locks.json"}
                                 and not p.name.startswith(("paired-locks.attempt", ".")) for p in out.iterdir())
    if dirty and not (owned and Path(owned).resolve() == out and not (out / "window7.json").exists()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    window7(args)


if __name__ == "__main__":
    main()
