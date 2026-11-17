#!/usr/bin/env python3
"""Amoeba M0: how much of a donor's activation "cone" must be copied for a faithful answer.

CPU only (torch, fp32). No GPU, no server. Research probe for
docs/research/2026-11-11-amoeba-rnn-plan.md (M0, cone width).

Method (subspace resample patching, joint across layers):
  * For each prompt, a *source* prompt of the same token length from the same set
    (capitals: another country; control: another TinyStories snippet).
  * At the output of every decoder layer l, at every position p >= 1, replace
      h <- P_U h + (I - P_U) h_src
    where U is the top-r directions of a per-layer basis. Position 0 (the attention
    sink, with massive activations) is always kept exact; this is recorded.
  * Positions further back than `reach` from the end are replaced entirely by the
    source (reach-back sweep); reach=None keeps all positions.
  * Faithfulness at the final position vs the clean run: KL(clean || patched),
    top-1 agreement, and (capitals) whether top-1 is the correct answer's first token.

Bases (fitted on a fit split, evaluated on a held-out split):
  * random  : random orthonormal basis (control for "any r dims").
  * pca     : centred PCA of clean residuals (variance ordering).
  * dpca    : PCA of the clean-minus-source differences dh (directions where prompts
              actually differ from their counterfactual).
  * attr    : eigenvectors of K = E[sym(g dh^T)], g = d logp(clean top1)/dh, dh = h - h_src,
              ranked by |eigenvalue|. u^T K u is the expected first-order effect carried
              along u, so this orders directions by how much answer they carry.

Sweeps:
  joint   : the same r at all layers at once (errors compound, which is the honest case).
  single  : r at one layer only, all other layers exact (per-layer width profile).
  reach   : full-rank but only the last k positions kept (how far back the cone reaches).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
    ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Brazil", "Bras"), ("Argentina", "Buenos"),
    ("Mexico", "Mexico"), ("India", "New"), ("Kenya", "Nairobi"), ("Greece", "Athens"),
    ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Ireland", "Dublin"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Turkey", "Ankara"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"), ("Peru", "Lima"), ("Chile", "Santiago"), ("Colombia", "Bog"),
    ("Cuba", "Havana"), ("Ukraine", "Kyiv"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
    ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Czechia", "Prague"), ("Switzerland", "Bern"),
    ("Morocco", "Rabat"), ("Ethiopia", "Addis"), ("Nigeria", "Abuja"), ("Ghana", "Accra"),
    ("Pakistan", "Islamabad"), ("Afghanistan", "Kabul"), ("Nepal", "Kathmandu"), ("Mongolia", "Ul"),
    ("Philippines", "Manila"), ("Malaysia", "Kuala"), ("Venezuela", "Caracas"), ("Uruguay", "Montevideo"),
    ("Israel", "Jerusalem"), ("Jordan", "Amman"), ("Syria", "Damascus"), ("Lebanon", "Beirut"),
    ("Iceland", "Reykjavik"), ("Estonia", "Tallinn"), ("Latvia", "Riga"), ("Lithuania", "Vilnius"),
]
TEMPLATE = "The capital of {} is"


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


# ---------------------------------------------------------------- prompt sets

def capitals_set(tok, n_max: int):
    items = []
    for country, cap in CAPITALS[:n_max]:
        ids = tok(TEMPLATE.format(country), add_special_tokens=False)["input_ids"]
        ans = tok(" " + cap, add_special_tokens=False)["input_ids"][0]
        items.append({"ids": ids, "answer": ans, "label": country})
    return items


def control_set(tok, n: int, length: int, seed: int, texts_json: str | None):
    if texts_json:  # pre-extracted {"source": parquet name, "texts": [...]} (venv lacks pyarrow)
        blob = json.loads(Path(texts_json).read_text())
        texts, path = list(blob["texts"]), blob["source"]
    else:
        import pyarrow.parquet as pq
        path = glob.glob(os.path.join(
            os.path.expanduser("~"),
            ".cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/*/data/validation-*.parquet"))[0]
        texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    rng = random.Random(seed)
    rng.shuffle(texts)
    items = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > length + 1:
            items.append({"ids": ids[:length], "answer": ids[length], "label": t[:40]})
        if len(items) >= n:
            break
    return items, path


NAMES = ("Lily Tom Ben Sue Max Mia Tim Anna Sam Lucy Jack Sara Bob Amy Joe Kate Mary John "
         "Jane Mark Emma Leo Zoe Ann").split()


def minpair_set(tok, n: int, length: int, seed: int, texts_json: str):
    """TinyStories snippets whose counterfactual swaps one character name (every occurrence).

    The fair control for capitals: the source differs in one word, not in the whole story.
    """
    texts = list(json.loads(Path(texts_json).read_text())["texts"])
    rng = random.Random(seed + 7)
    rng.shuffle(texts)
    name_ids = {}
    for nm in NAMES:
        ids = tok(" " + nm, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            name_ids[ids[0]] = nm
    items = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) <= length + 1:
            continue
        ids, ans = ids[:length], ids[length]
        hits = [i for i in ids[2:length - 4] if i in name_ids]
        if not hits:
            continue
        old = hits[0]
        new = rng.choice([k for k in name_ids if k != old and k not in ids])
        items.append({"ids": ids, "answer": ans, "src_ids": [new if i == old else i for i in ids],
                      "label": f"{name_ids[old]}->{name_ids[new]}: {t[:40]}"})
        if len(items) >= n:
            break
    return items


def effective_minpairs(model, items, n):
    """Keep name swaps that change the clean top-1 next token (else nothing needs copying)."""
    out = []
    for it in items:
        with torch.no_grad():
            a = model(input_ids=torch.tensor([it["ids"]])).logits[0, -1].argmax().item()
            b = model(input_ids=torch.tensor([it["src_ids"]])).logits[0, -1].argmax().item()
        if a != b:
            out.append(it)
        if len(out) >= n:
            break
    return out


def src_of(items, i):
    return items[i]["src_ids"] if "src_ids" in items[i] else items[items[i]["src"]]["ids"]


def pair_sources(items, seed: int):
    """Derangement within equal-length groups; drop items in singleton groups.

    Items that carry their own counterfactual (``src_ids``) are kept as they are.
    """
    if items and all("src_ids" in it for it in items):
        return list(range(len(items)))
    rng = random.Random(seed)
    groups: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        groups.setdefault(len(it["ids"]), []).append(i)
    keep = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        rng.shuffle(idx)  # shuffle order, then rotate so no fixed points
        perm = idx[1:] + idx[:1]
        for a, b in zip(idx, perm):
            items[a]["src"] = b
            keep.append(a)
    return sorted(keep)


# ---------------------------------------------------------------- model plumbing

class Patcher:
    """Forward hooks on decoder layers implementing subspace resample patching."""

    def __init__(self, model):
        self.layers = model.model.layers
        self.cfg = None  # dict(layer -> (Q[d,r] or None for full, src[B,T,d])) plus reach
        self.capture = None  # list to fill with layer outputs
        self.handles = [l.register_forward_hook(self._hook(i)) for i, l in enumerate(self.layers)]

    def _hook(self, i):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.capture is not None:
                if self.capture_grad:
                    h.retain_grad()
                self.capture[i] = h
            if self.cfg is not None and i in self.cfg["layers"]:
                Q = self.cfg["layers"][i]
                src = self.cfg["src"][i]
                T = h.shape[1]
                new = h.clone()
                if Q is None:  # full rank inside reach
                    body = h[:, 1:]
                elif Q.shape[1] == 0:
                    body = src[:, 1:]
                else:
                    x, s = h[:, 1:], src[:, 1:]
                    body = s + ((x - s) @ Q) @ Q.T
                new[:, 1:] = body
                reach = self.cfg.get("reach")
                if reach is not None and T - reach > 1:
                    new[:, 1:T - reach] = src[:, 1:T - reach]
                h = new
                if isinstance(out, tuple):
                    return (h,) + tuple(out[1:])
                return h
            return None
        return fn


def run(model, patcher, ids, cfg=None, capture=False, grad=False):
    patcher.cfg = cfg
    patcher.capture = {} if capture else None
    patcher.capture_grad = grad
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if grad:  # weights are frozen, so make the embeddings the leaf that carries the graph
            emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
            logits = model(inputs_embeds=emb).logits[:, -1].float()
        else:
            logits = model(input_ids=ids).logits[:, -1].float()
    caps = patcher.capture
    patcher.cfg, patcher.capture = None, None
    return logits, caps


def batches(items, idxs):
    """Group indices by length so each batch is rectangular."""
    by_len: dict[int, list[int]] = {}
    for i in idxs:
        by_len.setdefault(len(items[i]["ids"]), []).append(i)
    return list(by_len.values())


# ---------------------------------------------------------------- bases

def fit_bases(model, patcher, items, fit_idx, n_layers, d, seed):
    """Returns {basis: [Q_l (d x d) ordered columns]} fitted on fit_idx."""
    Ksum = [torch.zeros(d, d) for _ in range(n_layers)]
    Dsum = [torch.zeros(d, d) for _ in range(n_layers)]
    Xs = [[] for _ in range(n_layers)]
    count = 0
    for grp in batches(items, fit_idx):
        ids = torch.tensor([items[i]["ids"] for i in grp])
        sids = torch.tensor([src_of(items, i) for i in grp])
        _, scap = run(model, patcher, sids, capture=True)
        logits, cap = run(model, patcher, ids, capture=True, grad=True)
        top = logits.argmax(-1)
        lp = torch.log_softmax(logits, -1).gather(1, top[:, None]).sum()
        model.zero_grad(set_to_none=True)
        lp.backward()
        for l in range(n_layers):
            h = cap[l].detach()[:, 1:].reshape(-1, d)
            g = cap[l].grad[:, 1:].reshape(-1, d)
            dh = h - scap[l][:, 1:].reshape(-1, d)
            Ksum[l] += 0.5 * (g.T @ dh + dh.T @ g)
            Dsum[l] += dh.T @ dh
            Xs[l].append(h)
        count += len(grp)
    gen = torch.Generator().manual_seed(seed)
    bases = {"attr": [], "dpca": [], "pca": [], "random": []}
    spectra = {"attr": [], "dpca": [], "pca": []}
    for l in range(n_layers):
        K = Ksum[l] / count
        w, V = torch.linalg.eigh(K)
        order = torch.argsort(w.abs(), descending=True)
        bases["attr"].append(V[:, order].contiguous())
        spectra["attr"].append(w[order].abs().tolist())
        wd, Vd = torch.linalg.eigh(Dsum[l] / count)
        od = torch.argsort(wd, descending=True)
        bases["dpca"].append(Vd[:, od].contiguous())
        spectra["dpca"].append(wd[od].clamp_min(0).tolist())
        X = torch.cat(Xs[l])
        X = X - X.mean(0, keepdim=True)
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        P = Vh.T
        if P.shape[1] < d:  # pad with an orthonormal complement
            R = torch.randn(d, d - P.shape[1], generator=gen)
            R = R - P @ (P.T @ R)
            Qc, _ = torch.linalg.qr(R)
            P = torch.cat([P, Qc], 1)
        bases["pca"].append(P.contiguous())
        spectra["pca"].append((S ** 2).tolist())
        Qr, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
        bases["random"].append(Qr.contiguous())
    return bases, spectra


# ---------------------------------------------------------------- evaluation

def evaluate(model, patcher, items, idxs, layer_Q, reach, clean_cache, src_cache):
    """layer_Q: {layer: Q or None}. Returns metrics averaged over idxs."""
    kls, agree, correct, n = 0.0, 0, 0, 0
    for gk, grp in enumerate(batches(items, idxs)):
        ids = torch.tensor([items[i]["ids"] for i in grp])
        cfg = {"layers": layer_Q, "src": src_cache[gk], "reach": reach}
        logits, _ = run(model, patcher, ids, cfg=cfg)
        clean = clean_cache[gk]
        lc, lpch = torch.log_softmax(clean, -1), torch.log_softmax(logits, -1)
        kls += (lc.exp() * (lc - lpch)).sum(-1).sum().item()
        agree += (logits.argmax(-1) == clean.argmax(-1)).sum().item()
        ans = torch.tensor([items[i]["answer"] for i in grp])
        correct += (logits.argmax(-1) == ans).sum().item()
        n += len(grp)
    return {"kl": kls / n, "top1_agree": agree / n, "answer_acc": correct / n, "n": n}


def caches(model, patcher, items, idxs):
    clean, src = [], []
    for grp in batches(items, idxs):
        ids = torch.tensor([items[i]["ids"] for i in grp])
        sids = torch.tensor([src_of(items, i) for i in grp])
        lg, _ = run(model, patcher, ids)
        _, scap = run(model, patcher, sids, capture=True)
        clean.append(lg)
        src.append(scap)
    return clean, src


def min_r(rows, kl_tau, agree_tau):
    ok = [r for r in rows if r["kl"] <= kl_tau and r["top1_agree"] >= agree_tau]
    return min((r["r"] for r in ok), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-capitals", type=int, default=64)
    ap.add_argument("--n-control", type=int, default=48)
    ap.add_argument("--control-len", type=int, default=24)
    ap.add_argument("--ranks", default="0,1,2,4,8,16,32,64,128,256,384,512,768,1024")
    ap.add_argument("--single-ranks", default="1,4,16,64,256,512")
    ap.add_argument("--single-every", type=int, default=2, help="single-layer sweep stride")
    ap.add_argument("--reach", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--kl-tau", type=float, default=0.1)
    ap.add_argument("--agree-tau", type=float, default=0.9)
    ap.add_argument("--control-texts", default=None, help="JSON of pre-extracted TinyStories validation texts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sets", default="capitals,control")
    ap.add_argument("--in-sample", action="store_true",
                    help="evaluate on the fit split (tests whether width is per-instance, not a shared subspace)")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-reach", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = snapshot_download(args.model, local_files_only=True)
    weights = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    fingerprint = os.path.basename(snap)[:8]
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    patcher = Patcher(model)

    ranks = [int(x) for x in args.ranks.split(",")]
    single_ranks = [int(x) for x in args.single_ranks.split(",")]
    reaches = [int(x) for x in args.reach.split(",")]
    if args.smoke:
        args.n_capitals, args.n_control = 16, 12
        ranks, single_ranks, reaches = [0, 16, 1024], [16], [1, 4]
        args.single_every = 14

    ctl_items, ctl_path = control_set(tok, args.n_control, args.control_len, args.seed, args.control_texts)
    sets = {"capitals": capitals_set(tok, args.n_capitals), "control": ctl_items}
    if "minpair" in args.sets.split(","):
        sets["minpair"] = minpair_set(tok, args.n_control, args.control_len, args.seed, args.control_texts)
    t0 = time.time()
    report = {
        "checkpoint": f"{args.model}@{fingerprint}",
        "weights_sha256_12": {os.path.basename(w): sha(w) for w in weights},
        "mlx2_source_revision": git_rev(),
        "control_data": f"roneneldan/TinyStories validation ({os.path.basename(ctl_path)})",
        "device": "cpu", "dtype": "float32", "torch": torch.__version__,
        "args": vars(args), "position0_always_exact": True,
        "thresholds": {"kl_tau": args.kl_tau, "agree_tau": args.agree_tau},
        "sets": {},
    }

    report["eval_split"] = "fit (in-sample)" if args.in_sample else "held-out"
    for name, items in sets.items():
        if name not in args.sets.split(","):
            continue
        keep = pair_sources(items, args.seed)
        rng = random.Random(args.seed + 1)
        rng.shuffle(keep)
        half = len(keep) // 2
        fit_idx, ev_idx = sorted(keep[:half]), sorted(keep[half:])
        if args.in_sample:
            ev_idx = fit_idx
        clean, src = caches(model, patcher, items, ev_idx)
        clean_acc = sum(int(c.argmax(-1)[j] == items[i]["answer"])
                        for c, grp in zip(clean, batches(items, ev_idx))
                        for j, i in enumerate(grp)) / len(ev_idx)
        print(f"[{name}] fit={len(fit_idx)} eval={len(ev_idx)} clean answer acc={clean_acc:.3f}", flush=True)
        bases, spectra = fit_bases(model, patcher, items, fit_idx, n_layers, d, args.seed)
        res = {"n_fit": len(fit_idx), "n_eval": len(ev_idx), "clean_answer_acc": clean_acc,
               "eval_labels": [items[i]["label"] for i in ev_idx],
               "joint": {}, "single": {}, "reach": [], "spectra_top32": {
                   k: [s[:32] for s in v] for k, v in spectra.items()}}
        for bname, B in bases.items():
            rows = []
            for r in ranks:
                Q = {l: (None if r >= d else B[l][:, :r]) for l in range(n_layers)}
                m = evaluate(model, patcher, items, ev_idx, Q, None, clean, src)
                m["r"] = r
                rows.append(m)
                print(f"  joint {bname:6s} r={r:5d} kl={m['kl']:.4f} agree={m['top1_agree']:.3f} "
                      f"acc={m['answer_acc']:.3f}", flush=True)
            res["joint"][bname] = {"rows": rows, "min_r": min_r(rows, args.kl_tau, args.agree_tau)}
        for bname in (() if args.skip_single else ("attr", "dpca", "pca")):
            B = bases[bname]
            prof = {}
            for l in range(0, n_layers, args.single_every):
                rows = []
                for r in single_ranks + [0]:
                    m = evaluate(model, patcher, items, ev_idx, {l: B[l][:, :r]}, None, clean, src)
                    m["r"] = r
                    rows.append(m)
                prof[l] = {"rows": rows, "min_r": min_r(rows, args.kl_tau, args.agree_tau)}
                print(f"  single {bname:6s} L{l:02d} min_r={prof[l]['min_r']} "
                      + " ".join(f"r{x['r']}:{x['kl']:.3f}" for x in rows), flush=True)
            res["single"][bname] = prof
        for k in ([] if args.skip_reach else reaches):
            Q = {l: None for l in range(n_layers)}
            m = evaluate(model, patcher, items, ev_idx, Q, k, clean, src)
            m["reach"] = k
            res["reach"].append(m)
            print(f"  reach k={k:3d} kl={m['kl']:.4f} agree={m['top1_agree']:.3f} "
                  f"acc={m['answer_acc']:.3f}", flush=True)
        report["sets"][name] = res

    report["wall_s"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = round(rss_gb(), 2)
    tmp = out / "cone_width.json.tmp"
    tmp.write_text(json.dumps(report, indent=1))
    tmp.rename(out / "cone_width.json")
    print(f"wrote {out/'cone_width.json'} in {report['wall_s']}s, peak RSS {report['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
