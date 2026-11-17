"""Amoeba on a real host: teach frozen Qwen3-4B a skill it provably lacks (fict.atlas).

window5:
  A. Specialist: fine-tune Qwen3-0.6B's MLPs at layers 6-18 on the fictional atlas
     (plus general-text replay). Attention and every other weight stay at base.
  B. Baselines on the atlas probes (held-out phrasings): base 0.6B, specialist, base 4B.
  C. The box is the specialist's skill delta, a frozen donor sub-circuit per layer:
         Delta_l(z) = MLP_ft,l(z) - MLP_base,l(z)          (donor weights, token-local)
     wired into the frozen host at the depth-matched layer l':
         y_host += alpha * M_l Delta_l(R_l x_host)
     R_l (host MLP input -> donor MLP input) and M_l (donor residual -> host residual)
     are ridge stitches fitted on GENERIC English only. No atlas text touches the
     adapters, so any above-chance atlas accuracy in the host comes from the donor core.
     alpha is chosen on half the countries and reported on the other half.
  D. Controls: identity at alpha = 0 (must be exact); a random core (base MLP plus a
     random perturbation with the same per-matrix norm as the real delta); and the
     host's real-world capitals and off-topic NLL with the box on.
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

HERE = Path(__file__).resolve().parent
BOX_LAYERS = list(range(6, 19))
MLP_KEYS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def atlas():
    return json.loads((HERE / "topics" / "fict_atlas.json").read_text())


def probes(spec, tok, relation, countries=None):
    cands = spec["candidates"][relation]
    key = "capital_token" if relation == "capital" else "currency_token"
    out = []
    for it in spec["items"]:
        if countries is not None and it["country"] not in countries:
            continue
        for t in spec["probe_templates"][relation]:
            out.append((tok(t.format(country=it["country"]), add_special_tokens=False)["input_ids"], it[key], cands))
    return out


def atlas_stream(spec, tok, T, seed=0):
    rng = random.Random(seed)
    items, temps = spec["items"], spec["train_templates"]

    def para():
        ids = []
        while len(ids) < T + 1:
            it = rng.choice(items)
            ids += tok(" " + rng.choice(temps).format(**it), add_special_tokens=False)["input_ids"]
        return ids[:T + 1]
    return para


def general_stream(tok, T, seed=0):
    rng = random.Random(seed)
    buf = []
    for name, k in (("tinystories_train.jsonl", 4000), ("billsum.jsonl", 800), ("scifact.jsonl", 1500)):
        for l in (DATA / name).read_text().splitlines()[:k]:
            buf += tok(json.loads(l)["text"], add_special_tokens=False)["input_ids"] + [151643]
    arr = np.array(buf, dtype=np.int32)

    def win():
        o = rng.randrange(0, len(arr) - T - 2)
        return arr[o:o + T + 1].tolist()
    return win


# ------------------------------------------------------------------ A. specialist
def finetune(donor, spec, tok, steps, B, lr, T=128, log=print):
    base = {f"model.layers.{l}.{k}": donor.w[f"model.layers.{l}.{k}"] for l in BOX_LAYERS for k in MLP_KEYS}
    params = {k: mx.array(v) for k, v in base.items()}
    para, gen = atlas_stream(spec, tok, T), general_stream(tok, T)
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(p, ids):
        saved = {k: donor.w[k] for k in p}
        donor.w.update(p)
        logits, _, _ = donor.forward(ids[:, :-1])
        donor.w.update(saved)
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
        if s % 200 == 0 or s == steps - 1:
            hist.append({"step": s, "sec": round(time.time() - t0, 1), "loss": float(loss)}); log(hist[-1])
    delta_norms = {k: float(mx.linalg.norm(params[k] - base[k]) / mx.linalg.norm(base[k])) for k in params}
    return params, base, hist, delta_norms


def with_weights(donor, weights):
    class Ctx:
        def __enter__(self):
            self.saved = {k: donor.w[k] for k in weights}; donor.w.update(weights)

        def __exit__(self, *a):
            donor.w.update(self.saved)
    return Ctx()


# ------------------------------------------------------------------ C. stitching
def layer_map(Ld, Lh):
    return {l: int(round((l + 0.5) * Lh / Ld - 0.5)) for l in BOX_LAYERS}


def fit_stitches(donor, host, tok, lmap, n_seq, T=256, lam=1e-2):
    """Streaming ridge: R_l host x_{l'} -> donor x_l (MLP inputs); M_l donor residual h_l -> host residual h_{l'}."""
    gen = general_stream(tok, T, seed=77)
    stats = {l: {"XtX": 0, "XtY": 0, "DtD": 0, "DtH": 0} for l in BOX_LAYERS}
    rec_d = set(BOX_LAYERS); rec_h = set(lmap.values())
    for i in range(0, n_seq, 8):
        ids = mx.array([gen()[:T] for _ in range(8)])
        _, capd, _ = donor.forward(ids, record=rec_d, stop_after=max(rec_d))
        _, caph, _ = host.forward(ids, record=rec_h, stop_after=max(rec_h))
        resd = residuals(donor, ids, rec_d); resh = residuals(host, ids, rec_h)
        for l in BOX_LAYERS:
            lh = lmap[l]
            X = caph[lh][0].reshape(-1, host.d).astype(mx.float32); Y = capd[l][0].reshape(-1, donor.d).astype(mx.float32)
            D = resd[l].reshape(-1, donor.d).astype(mx.float32); Hh = resh[lh].reshape(-1, host.d).astype(mx.float32)
            one = mx.ones((X.shape[0], 1))
            X = mx.concatenate([X, one], 1); D = mx.concatenate([D, one], 1)
            s = stats[l]
            s["XtX"] = s["XtX"] + X.T @ X; s["XtY"] = s["XtY"] + X.T @ Y
            s["DtD"] = s["DtD"] + D.T @ D; s["DtH"] = s["DtH"] + D.T @ Hh
        mx.eval([list(v.values()) for v in stats.values()])
    st = {}
    for l, s in stats.items():
        n = float(s["XtX"][-1, -1])
        R = mx.linalg.solve(s["XtX"] + lam * n * mx.eye(s["XtX"].shape[0]), s["XtY"], stream=mx.cpu)
        M = mx.linalg.solve(s["DtD"] + lam * n * mx.eye(s["DtD"].shape[0]), s["DtH"], stream=mx.cpu)
        st[l] = {"R": R, "M": M[:-1]}      # M applied to a residual *delta* drops the intercept row
    mx.eval([list(v.values()) for v in st.values()])
    return st


def residuals(model, ids, layers):
    """Residual stream right after each requested layer (pre-norm stream)."""
    out = {}
    h = model.embed[ids]
    for i in range(max(layers) + 1):
        a, _ = model.attention(i, model.norm(h, model.p(i, "input_layernorm.weight")))
        h = h + a
        x = model.norm(h, model.p(i, "post_attention_layernorm.weight"))
        h = h + model.mlp(i, x)
        if i in layers:
            out[i] = h
    return out


def stitch_quality(donor, host, tok, lmap, st, n_seq=16, T=256):
    gen = general_stream(tok, T, seed=991)
    ids = mx.array([gen()[:T] for _ in range(n_seq)])
    _, capd, _ = donor.forward(ids, record=set(BOX_LAYERS), stop_after=max(BOX_LAYERS))
    _, caph, _ = host.forward(ids, record=set(lmap.values()), stop_after=max(lmap.values()))
    out = {}
    for l in BOX_LAYERS:
        X = caph[lmap[l]][0].reshape(-1, host.d).astype(mx.float32); Y = capd[l][0].reshape(-1, donor.d)
        P = mx.concatenate([X, mx.ones((X.shape[0], 1))], 1) @ st[l]["R"]
        out[l] = float(1 - mx.sum((Y - P) ** 2) / mx.sum((Y - mx.mean(Y, 0)) ** 2))
    return out


def box_hook(donor, host, lmap, st, ft, base, alpha):
    inv = {lmap[l]: l for l in BOX_LAYERS}

    def mlp_with(i, z, weights):
        g = z @ weights[f"model.layers.{i}.mlp.gate_proj.weight"].T
        u = z @ weights[f"model.layers.{i}.mlp.up_proj.weight"].T
        return (g * mx.sigmoid(g) * u) @ weights[f"model.layers.{i}.mlp.down_proj.weight"].T

    def hook(i, x, y):
        if alpha == 0 or i not in inv:
            return None
        l = inv[i]
        X = x.astype(mx.float32)
        z = mx.concatenate([X, mx.ones(X.shape[:-1] + (1,))], -1) @ st[l]["R"]
        delta = mlp_with(l, z, ft) - mlp_with(l, z, base)
        return y + (alpha * (delta @ st[l]["M"])).astype(y.dtype)
    return hook


def random_core(ft, base, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ft:
        d = ft[k] - base[k]
        r = mx.array(rng.standard_normal(d.shape).astype(np.float32))
        out[k] = base[k] + r * (mx.linalg.norm(d) / mx.linalg.norm(r))
    return out


# ------------------------------------------------------------------ window
def window5(args):
    out = Path(args.out); t0 = time.time()
    spec = atlas()
    donor = Donor("Qwen/Qwen3-0.6B")
    tok = tokenizer(donor)
    rec = {"schema": "amoeba.m2.window5.v1", "donor": donor.artifact, "source_revision": source_rev(),
           "skill": "fict.atlas v1", "box_layers": BOX_LAYERS, "started": t0}
    cap_all = probes(spec, tok, "capital"); cur_all = probes(spec, tok, "currency")
    off = evals.offtopic_texts(DATA, tok, n_seq=8 if args.smoke else 48)
    rec["chance"] = spec["chance"]
    rec["base_donor"] = {"capital": evals.topic_accuracy(donor, cap_all), "currency": evals.topic_accuracy(donor, cur_all),
                         "offtopic_nll": evals.offtopic_loss(donor, off)}
    print("base_donor", json.dumps({k: v["acc"] if isinstance(v, dict) else v for k, v in rec["base_donor"].items()}), flush=True)
    steps, B = (10, 4) if args.smoke else (args.ft_steps, 16)
    logf = open(out / "specialist_log.jsonl", "a")
    ft, base, hist, dn = finetune(donor, spec, tok, steps, B, args.ft_lr,
                                 log=lambda r: (logf.write(json.dumps(r) + "\n"), logf.flush(), print("ft", r, flush=True)))
    mx.save_safetensors(str(out / "specialist_mlp_layers6-18.safetensors"), ft)
    with with_weights(donor, ft):
        rec["specialist"] = {"capital": evals.topic_accuracy(donor, cap_all), "currency": evals.topic_accuracy(donor, cur_all),
                             "offtopic_nll": evals.offtopic_loss(donor, off),
                             "real_capitals_heldout": evals.topic_accuracy(donor, evals.topic_prompts(evals.load_topic(), "heldout", tok)),
                             "train_tokens": steps * B * 128, "loss_hist": hist,
                             "relative_delta_norm_mean": float(np.mean(list(dn.values())))}
    atomic(out / "window5.json", rec)
    print("specialist", json.dumps({k: (v["acc"] if isinstance(v, dict) and "acc" in v else v) for k, v in rec["specialist"].items() if k != "loss_hist"}), flush=True)
    # host
    host = Donor(args.host, dtype=mx.bfloat16 if args.host_bf16 else mx.float32)
    rec["host"] = host.artifact
    real = evals.topic_prompts(evals.load_topic(), "heldout", tok)
    rec["base_host"] = {"capital": evals.topic_accuracy(host, cap_all), "currency": evals.topic_accuracy(host, cur_all),
                        "offtopic_nll": evals.offtopic_loss(host, off), "real_capitals_heldout": evals.topic_accuracy(host, real)}
    atomic(out / "window5.json", rec)
    print("base_host", json.dumps({k: (v["acc"] if isinstance(v, dict) else v) for k, v in rec["base_host"].items()}), flush=True)
    lmap = layer_map(donor.L, host.L)
    st = fit_stitches(donor, host, tok, lmap, n_seq=16 if args.smoke else args.stitch_seqs)
    rec["layer_map"] = lmap
    rec["read_stitch_r2"] = stitch_quality(donor, host, tok, lmap, st, n_seq=4 if args.smoke else 16)
    print("stitch", json.dumps({k: round(v, 3) for k, v in rec["read_stitch_r2"].items()}), flush=True)
    # identity at alpha = 0
    ids = mx.array([cap_all[0][0]])
    l0, _, _ = host.forward(ids); l1, _, _ = host.forward(ids, mlp_hook=box_hook(donor, host, lmap, st, ft, base, 0.0))
    rec["identity_alpha0_max_abs_diff"] = float(mx.max(mx.abs(l0 - l1)))
    # alpha chosen on half the countries, reported on the other half
    names = [it["country"] for it in spec["items"]]
    rng = random.Random(5); rng.shuffle(names)
    sel, rep = set(names[:60]), set(names[60:])
    sweep = {}
    for a in ([1.0] if args.smoke else [0.5, 1.0, 2.0, 4.0, 8.0]):
        hk = box_hook(donor, host, lmap, st, ft, base, a)
        sweep[a] = evals.topic_accuracy(host, probes(spec, tok, "capital", sel), hook=hk)["acc"]
        print("sweep", a, sweep[a], flush=True)
    best = max(sweep, key=sweep.get)
    hk = box_hook(donor, host, lmap, st, ft, base, best)
    rc = random_core(ft, base)
    hk_r = box_hook(donor, host, lmap, st, rc, base, best)
    rec["transfer"] = {
        "alpha_sweep_selection_half": sweep, "alpha": best,
        "capital_report_half": evals.topic_accuracy(host, probes(spec, tok, "capital", rep), hook=hk),
        "currency_report_half": evals.topic_accuracy(host, probes(spec, tok, "currency", rep), hook=hk),
        "capital_report_half_no_box": evals.topic_accuracy(host, probes(spec, tok, "capital", rep)),
        "offtopic_nll_with_box": evals.offtopic_loss(host, off, hook=hk),
        "real_capitals_heldout_with_box": evals.topic_accuracy(host, real, hook=hk),
        "random_core": {"capital_report_half": evals.topic_accuracy(host, probes(spec, tok, "capital", rep), hook=hk_r),
                        "offtopic_nll": evals.offtopic_loss(host, off, hook=hk_r)},
    }
    rec["finished"] = time.time()
    atomic(out / "window5.json", rec)
    t = rec["transfer"]
    print("transfer", json.dumps({"alpha": best, "capital": t["capital_report_half"]["acc"], "no_box": t["capital_report_half_no_box"]["acc"],
                                  "currency": t["currency_report_half"]["acc"], "random_core": t["random_core"]["capital_report_half"]["acc"],
                                  "offtopic_with_box": t["offtopic_nll_with_box"], "offtopic_base": rec["base_host"]["offtopic_nll"],
                                  "real_caps_with_box": t["real_capitals_heldout_with_box"]["acc"],
                                  "identity": rec["identity_alpha0_max_abs_diff"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window5"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--host-bf16", action="store_true")
    ap.add_argument("--ft-steps", type=int, default=1500)
    ap.add_argument("--ft-lr", type=float, default=3e-5)
    ap.add_argument("--stitch-seqs", type=int, default=192)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    owned = os.environ.get("AMOEBA_OWNED_OUT")
    dirty = out.exists() and any(p.name not in {"desktop-lease.json", "paired-locks.json"}
                                 and not p.name.startswith(("paired-locks.attempt", ".")) for p in out.iterdir())
    if dirty and not (owned and Path(owned).resolve() == out and not (out / "window5.json").exists()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    window5(args)


if __name__ == "__main__":
    main()
