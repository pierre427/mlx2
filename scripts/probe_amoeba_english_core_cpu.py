#!/usr/bin/env python3
"""Amoeba M0b-core: is there an English *core* box, with knowledge boxes on top of it?

CPU only (torch, fp32). No GPU, no server, no activation capture to RAM (everything streams).

1. Core: one *shared* mask over MLP neurons (layer x neuron, the same at every position and for
   every text), learned on broad English (the M0a-3 corpus: TinyStories train, repo markdown,
   donor-generated encyclopedic text). Masked-off neurons are mean-ablated (replaced by their mean
   activation over a fit sample), so their content is removed, not swapped. Objective:
       KL(clean || core) per token, over all positions >= 1, + lam * mean(mask)
   warm-started across increasing lam, binarised and scored exactly on held-out text per domain
   (cross-entropy vs the true next token, KL, top-1 agreement) and on the 64 capitals prompts.
   Control: a random core with the same per-layer neuron counts.
2. Box on the core: with the core fixed, learn a capitals box = shared mask over the non-core
   neurons, on 32 fit capitals, tested on the 32 held-out capitals, plus its effect on English.
3. What the core keeps, drops and boxes (the expert-token-signature method, hidden-state version, per neuron):
   domain profile (mean |act - mean| per domain), top tokens it fires on, and for late layers the
   tokens its write direction promotes through the final norm + unembedding. Reported for core,
   dropped and box neurons against random neuron sets of the same size (the method's controls).
   Inside the core: token-category shares of top tokens (function words, punctuation, word pieces,
   numbers, content), cross-domain variation, and the most firmly kept neurons per depth band.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_amoeba_cone_width_cpu as cw  # noqa: E402

DOMAINS = ["tinystories_train", "repo_markdown", "donor_generated"]
FUNCTION_WORDS = set("""a an the and or but if then so because as of in on at to for from by with without
about into over under after before between through during up down out off than that this these those
it its he she they them his her their we us our you your i me my is are was were be been being am do
does did have has had will would can could shall should may might must not no yes there here what which
who whom whose when where why how all any some each every both either neither one very too also just
only even still again""".split())


def token_category(t: str) -> str:
    if t.strip() == "" or all(not c.isalnum() for c in t):
        return "punct_space"
    w = t.strip().lower()
    if t.startswith(" ") and w in FUNCTION_WORDS:
        return "function_word"
    if not t.startswith(" ") and t[:1].isalpha():
        return "word_piece"
    if any(c.isdigit() for c in t):
        return "number"
    return "content"


class MeanAblate:
    """MLP hook: act_used = mean + m * (act - mean); m is [L, N] (shared) or None (clean)."""

    def __init__(self, model):
        self.mlps = [l.mlp for l in model.model.layers]
        self.mask, self.mean, self.stats = None, None, None
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(self.mlps)]

    def _hook(self, l):
        def fn(mod, args, out):
            if self.mask is None and self.stats is None:
                return None
            x = args[0]
            act = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            if self.stats is not None:
                self.stats(l, act.detach())
            if self.mask is None:
                return None
            m = self.mask[l]
            used = self.mean[l] + m * (act - self.mean[l])
            used = torch.cat([act[:, :1], used[:, 1:]], 1)      # position 0 (sink) exact
            return mod.down_proj(used)
        return fn


def kl_rows(clean_logits, logits):
    lc, lp = torch.log_softmax(clean_logits, -1), torch.log_softmax(logits, -1)
    return (lc.exp() * (lc - lp)).sum(-1)


def score_text(model, H, seqs, mask, bs=8):
    """Per-token CE (true next token), KL vs clean and top-1 agreement, positions 1..T-2."""
    ce_c = ce_m = kl = ag = n = 0.0
    for i in range(0, len(seqs), bs):
        ids = torch.tensor(seqs[i:i + bs])
        with torch.no_grad():
            H.mask = None
            lc = model(input_ids=ids).logits[:, 1:-1].float()
            H.mask = mask
            lm = model(input_ids=ids).logits[:, 1:-1].float()
            H.mask = None
        tgt = ids[:, 2:]
        ce_c += torch.nn.functional.cross_entropy(lc.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        ce_m += torch.nn.functional.cross_entropy(lm.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        kl += kl_rows(lc, lm).sum().item()
        ag += (lc.argmax(-1) == lm.argmax(-1)).sum().item()
        n += tgt.numel()
    return {"ce_clean": ce_c / n, "ce_masked": ce_m / n, "d_ce": (ce_m - ce_c) / n,
            "kl": kl / n, "top1_agree": ag / n, "tokens": int(n)}


def score_capitals(model, H, items, idx, mask):
    acc_c = acc_m = kl = ag = 0.0
    for grp in cw.batches(items, idx):
        ids = torch.tensor([items[i]["ids"] for i in grp])
        ans = torch.tensor([items[i]["answer"] for i in grp])
        with torch.no_grad():
            H.mask = None
            lc = model(input_ids=ids).logits[:, -1].float()
            H.mask = mask
            lm = model(input_ids=ids).logits[:, -1].float()
            H.mask = None
        acc_c += (lc.argmax(-1) == ans).sum().item()
        acc_m += (lm.argmax(-1) == ans).sum().item()
        kl += kl_rows(lc, lm).sum().item()
        ag += (lc.argmax(-1) == lm.argmax(-1)).sum().item()
    n = len(idx)
    return {"acc_clean": acc_c / n, "acc_masked": acc_m / n, "kl": kl / n, "top1_agree": ag / n, "n": n}


ENTITY_TEMPLATES = ["The capital of {} is", "{} has its capital in", "In {}, the capital city is",
                    "The government of {} sits in the city of", "If you visit {}, its capital is"]


def entity_probe(model, H, tok, mask, countries, layers):
    """Is the entity decodable? Leave-one-template-out nearest-centroid over countries.

    Hidden states (decoder-layer outputs) at the last token and at the country's last
    sub-token, standardised over all samples; cosine to per-country centroids built from
    the other templates. Chance = 1 / len(countries).
    """
    feats = {(pos, l): [] for pos in ("last", "entity") for l in layers}
    caps = {}

    def grab(mod, args, out, l=None):
        caps[l] = (out[0] if isinstance(out, tuple) else out).detach()
    hooks = [model.model.layers[l].register_forward_hook(lambda m, a, o, l=l: grab(m, a, o, l)) for l in layers]
    H.mask = mask
    try:
        for t in ENTITY_TEMPLATES:
            row = {k: [] for k in feats}
            for c in countries:
                text = t.format(c)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                ent = len(tok(text[:text.index(c) + len(c)], add_special_tokens=False)["input_ids"]) - 1
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids]))
                for l in layers:
                    row[("last", l)].append(caps[l][0, -1])
                    row[("entity", l)].append(caps[l][0, ent])
            for k in feats:
                feats[k].append(torch.stack(row[k]))                  # C x d per template
    finally:
        H.mask = None
        for h in hooks:
            h.remove()
    out = {}
    nT, C = len(ENTITY_TEMPLATES), len(countries)
    for (pos, l), per_t in feats.items():
        X = torch.stack(per_t)                                        # T x C x d
        mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0).clamp_min(1e-6)
        X = torch.nn.functional.normalize((X - mu) / sd, dim=-1)
        hit = 0
        for t in range(nT):
            cent = torch.nn.functional.normalize(torch.cat([X[:t], X[t + 1:]]).mean(0), dim=-1)
            hit += int((X[t] @ cent.T).argmax(-1).eq(torch.arange(C)).sum())
        out[f"{pos}@L{l}"] = hit / (nT * C)
    out["chance"] = 1.0 / C
    return out


def mean_acts(model, H, seqs, L, N, bs=8):
    s = torch.zeros(L, N, dtype=torch.float64)
    cnt = [0]

    def acc(l, act):
        s[l] += act[:, 1:].reshape(-1, N).double().sum(0)
        if l == 0:
            cnt[0] += act[:, 1:].shape[0] * act[:, 1:].shape[1]
    H.stats = acc
    for i in range(0, len(seqs), bs):
        with torch.no_grad():
            model(input_ids=torch.tensor(seqs[i:i + bs]))
    H.stats = None
    return (s / cnt[0]).float()


def train_mask(model, H, batches_fn, Z, fixed, lam, steps, lr, loss_fn, log, tag, straight_through=False):
    """Generic shared-mask optimiser. Effective mask = fixed + (1 - fixed) * sigmoid(Z).

    straight_through: the forward pass uses the binary mask (Z > 0) and gradients flow through
    the sigmoid, so the optimised mask is the one that is scored. Without it a soft mask can lean
    on small leaks from every "off" unit, which vanish when it is binarised (seen on boxes that
    start near off).
    """
    opt = torch.optim.Adam([Z], lr=lr)
    free = 1.0 - fixed
    for step in range(steps):
        soft = torch.sigmoid(Z)
        if straight_through:
            soft = (Z > 0).float() + soft - soft.detach()
        m = fixed + free * soft
        loss_main = loss_fn(m, batches_fn(step))
        size = (free * torch.sigmoid(Z)).sum() / free.sum().clamp_min(1)
        loss = loss_main + lam * size
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            log(f"{time.strftime('%H:%M:%S')}   {tag} lam={lam} step {step} main={loss_main.item():.4f} "
                f"kept={float((fixed + free * (Z.detach() > 0).float()).mean()):.4f}")
    return Z


def signatures(model, H, seqs_by_domain, cap_items, mean, L, N, tok, top_k=8):
    """Streaming per-neuron hidden-state signatures (domain profile + top tokens)."""
    dom_names = list(seqs_by_domain) + ["capitals"]
    dsum = torch.zeros(len(dom_names), L, N)
    dcnt = torch.zeros(len(dom_names))
    topv = torch.full((L, N, top_k), -1.0)
    topt = torch.zeros((L, N, top_k), dtype=torch.long)
    state = {}

    def acc(l, act):
        dev = (act[:, 1:] - mean[l]).abs()                   # B,T-1,N
        di = state["d"]
        dsum[di, l] += dev.reshape(-1, N).sum(0)
        if l == 0:
            dcnt[di] += dev.shape[0] * dev.shape[1]
        flat = dev.reshape(-1, N)                             # tokens x N
        tk = state["tok"].reshape(-1)
        v, i = flat.topk(min(top_k, flat.shape[0]), dim=0)    # k x N
        cand_v = torch.cat([topv[l], v.T], 1)
        cand_t = torch.cat([topt[l], tk[i].T], 1)
        sv, si = cand_v.topk(top_k, dim=1)
        topv[l], topt[l] = sv, cand_t.gather(1, si)
    H.stats = acc
    for di, name in enumerate(dom_names):
        state["d"] = di
        if name == "capitals":
            batches = [[it["ids"] for it in cap_items if len(it["ids"]) == n]
                       for n in sorted({len(it["ids"]) for it in cap_items})]
        else:
            seqs = seqs_by_domain[name]
            batches = [seqs[i:i + 8] for i in range(0, len(seqs), 8)]
        for b in batches:
            ids = torch.tensor(b)
            state["tok"] = ids[:, 1:]
            with torch.no_grad():
                model(input_ids=ids)
    H.stats = None
    prof = dsum / dcnt[:, None, None]
    return dom_names, prof, topv, topt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True, help="corpus.json from the M0a-3 sparse-cut work dir")
    ap.add_argument("--n-fit", type=int, default=900)
    ap.add_argument("--n-mean", type=int, default=240)
    ap.add_argument("--n-heldout", type=int, default=60, help="held-out sequences per domain")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lambdas", default="0.25,0.5,1,2,4")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--core-dce", type=float, default=0.15,
                    help="pick the smallest core whose mean held-out d_CE <= this (nats/token)")
    ap.add_argument("--box-lambdas", default="0.25,1,4,16")
    ap.add_argument("--box-core-lams", default="", help="core lams to build boxes on (default: threshold pick)")
    ap.add_argument("--retokenize-from", default="", help="tokenizer the corpus ids belong to, if not --model's")
    ap.add_argument("--core-masks", default="", help="comma list of saved masks_core<lam>.pt: skip the core sweep")
    ap.add_argument("--entity-layers", default="8,14,20,27")
    ap.add_argument("--box-steps", type=int, default=120)
    ap.add_argument("--ablation", choices=["mean", "zero"], default="mean",
                    help="zero gives a mask that exports to an ordinary checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    logf = open(out / "run.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = snapshot_download(args.model, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L, N = model.config.num_hidden_layers, model.config.intermediate_size
    H = MeanAblate(model)
    t0 = time.time()
    report = {"checkpoint": f"{args.model}@{os.path.basename(snap)[:8]}",
              "weights_sha256_12": {os.path.basename(w): cw.sha(w)
                                    for w in sorted(glob.glob(os.path.join(snap, "*.safetensors")))},
              "mlx2_source_revision": cw.git_rev(), "device": "cpu", "dtype": "float32",
              "args": vars(args), "unit": f"MLP neurons, shared over positions ({L} x {N})",
              "ablation": f"{args.ablation} (mean = fit-sample mean activation); position 0 exact"}

    # ---- data: disjoint mean / fit / held-out splits per domain, sequences cut to seq_len
    corpus = json.loads(Path(args.corpus).read_text())["seqs"]
    if args.retokenize_from:        # corpus ids belong to another tokenizer: decode, re-encode
        src_tok = AutoTokenizer.from_pretrained(snapshot_download(args.retokenize_from, local_files_only=True))
        corpus = {d: [tok(src_tok.decode(q), add_special_tokens=False)["input_ids"] for q in corpus[d]]
                  for d in DOMAINS}
        corpus = {d: [q for q in v if len(q) >= args.seq_len] for d, v in corpus.items()}
        report["retokenized_from"] = args.retokenize_from
    split = {}
    for d in DOMAINS:
        s = [q[:args.seq_len] for q in corpus[d]]
        rng.shuffle(s)
        k_fit = args.n_fit // len(DOMAINS)
        k_mean = args.n_mean // len(DOMAINS)
        split[d] = {"held": s[:args.n_heldout], "mean": s[args.n_heldout:args.n_heldout + k_mean],
                    "fit": s[args.n_heldout + k_mean:args.n_heldout + k_mean + k_fit]}
    fit = [q for d in DOMAINS for q in split[d]["fit"]]
    rng.shuffle(fit)
    mean = mean_acts(model, H, [q for d in DOMAINS for q in split[d]["mean"]], L, N)
    if args.ablation == "zero":     # exportable: dropped neurons become zero output columns
        mean = torch.zeros_like(mean)
    H.mean = mean
    log(f"{time.strftime('%H:%M:%S')} means over {sum(len(split[d]['mean']) for d in DOMAINS)} seqs; fit={len(fit)}")

    caps = cw.capitals_set(tok, 64)
    order = list(range(len(caps)))
    random.Random(args.seed + 1).shuffle(order)
    cap_fit, cap_held = sorted(order[:32]), sorted(order[32:])

    def held_scores(mask):
        r = {d: score_text(model, H, split[d]["held"], mask) for d in DOMAINS}
        r["mean_d_ce"] = sum(r[d]["d_ce"] for d in DOMAINS) / len(DOMAINS)
        r["capitals_heldout"] = score_capitals(model, H, caps, cap_held, mask)
        r["capitals_fit"] = score_capitals(model, H, caps, cap_fit, mask)
        r["capitals_all"] = score_capitals(model, H, caps, list(range(len(caps))), mask)
        return r

    # ---- 1. English core
    def fit_batches(step):
        i = (step * args.bs) % len(fit)
        return torch.tensor((fit + fit)[i:i + args.bs])

    def core_loss(m, ids):
        with torch.no_grad():
            H.mask = None
            lc = model(input_ids=ids).logits[:, 1:].float()
        H.mask = m
        lm = model(input_ids=ids).logits[:, 1:].float()
        H.mask = None
        return kl_rows(lc, lm).mean()

    Z = torch.full((L, N), 3.0, requires_grad=True)
    zero = torch.zeros(L, N)
    cores = []
    if args.core_masks:              # box stage only, on cores saved by an earlier run
        lams = []
        for path in args.core_masks.split(","):
            blob = torch.load(path)
            if not torch.equal(blob["mean"], mean):
                log(f"note: {path} was learned with a different mean; using its own")
            lam = float(Path(path).stem.replace("masks_core", ""))
            H.mean = mean = blob["mean"]
            M = blob["core"]
            row = {"lam": lam, "kept_frac": float(M.mean()), "kept_per_layer": M.sum(1).tolist(),
                   "learned": held_scores(M), "source_mask": path}
            cores.append((row, M.clone()))
            lams.append(str(lam))
            log(f"[core loaded] lam={lam} kept={row['kept_frac']:.3f} d_CE={row['learned']['mean_d_ce']:.3f} "
                f"cap_acc={row['learned']['capitals_all']['acc_masked']:.2f}")
        args.box_core_lams = ",".join(lams)
        args.lambdas = ""
    ref = held_scores(torch.zeros(L, N))   # everything mean-ablated: the floor
    report["all_ablated"] = ref
    log(f"all-MLP-ablated: mean d_CE={ref['mean_d_ce']:.3f} capitals acc={ref['capitals_all']['acc_masked']:.3f}")
    for lam in [float(x) for x in args.lambdas.split(",") if x]:
        Z = train_mask(model, H, fit_batches, Z, zero, lam, args.steps, args.lr, core_loss, log, "core")
        M = (Z.detach() > 0).float()
        per_layer = M.sum(1)
        rnd = torch.zeros(L, N)
        g = torch.Generator().manual_seed(args.seed + int(lam * 100))
        for l in range(L):
            k = int(per_layer[l])
            if k:
                rnd[l, torch.randperm(N, generator=g)[:k]] = 1.0
        row = {"lam": lam, "kept_frac": float(M.mean()), "kept_per_layer": per_layer.tolist(),
               "learned": held_scores(M), "random_same_counts": held_scores(rnd)}
        cores.append((row, M.clone()))
        per_domain = ", ".join("%s:%.3f" % (d[:4], row["learned"][d]["d_ce"]) for d in DOMAINS)
        log(f"[core] lam={lam} kept={row['kept_frac']:.3f} learned d_CE={row['learned']['mean_d_ce']:.3f} "
            f"({per_domain}) "
            f"cap_acc={row['learned']['capitals_all']['acc_masked']:.2f} | random d_CE="
            f"{row['random_same_counts']['mean_d_ce']:.3f} cap_acc={row['random_same_counts']['capitals_all']['acc_masked']:.2f}")
    report["core_sweep"] = [r for r, _ in cores]
    if args.box_core_lams:
        want = [float(x) for x in args.box_core_lams.split(",")]
        picks = [(r, M) for r, M in cores if any(abs(r["lam"] - w) < 1e-9 for w in want)]
    else:
        ok = [(r, M) for r, M in cores if r["learned"]["mean_d_ce"] <= args.core_dce]
        picks = [min(ok, key=lambda x: x[0]["kept_frac"]) if ok else min(cores, key=lambda x: x[0]["learned"]["mean_d_ce"])]
    log("box cores: " + ", ".join(f"lam={r['lam']} kept={r['kept_frac']:.3f}" for r, _ in picks))

    cap_groups = cw.batches(caps, cap_fit)
    clean_cap = {}
    for gi, grp in enumerate(cap_groups):
        with torch.no_grad():
            H.mask = None
            clean_cap[gi] = model(input_ids=torch.tensor([caps[i]["ids"] for i in grp])).logits[:, -1].float()

    def box_loss(m, _):
        tot = 0.0
        for gi, grp in enumerate(cap_groups):
            H.mask = m
            lm = model(input_ids=torch.tensor([caps[i]["ids"] for i in grp])).logits[:, -1].float()
            H.mask = None
            tot = tot + kl_rows(clean_cap[gi], lm).sum()
        return tot / len(cap_fit)

    # signatures are mask-independent: compute once, then read them per (core, box) grouping
    sig_domains = {d: split[d]["held"] for d in DOMAINS}
    dom_names, prof, topv, topt = signatures(model, H, sig_domains, caps, mean, L, N, tok)
    sel = prof / prof.mean(0, keepdim=True).clamp_min(1e-8)
    cv = prof.std(0) / prof.mean(0).clamp_min(1e-8)
    W_U = model.lm_head.weight.detach()
    norm_w = model.model.norm.weight.detach()
    ci = dom_names.index("capitals")
    cats = ["function_word", "punct_space", "word_piece", "number", "content"]
    cat_of = {}

    def cat_share(mask):
        cnt = {c: 0 for c in cats}
        tot = 0
        for l, n in mask.nonzero().tolist():
            for t in topt[l, n].tolist():
                c = cat_of.get(t)
                if c is None:
                    c = cat_of[t] = token_category(tok.decode([t]))
                cnt[c] += 1
                tot += 1
        return {c: cnt[c] / max(1, tot) for c in cats}

    def describe(l, n, extra=None):
        item = {"layer": l, "neuron": n, "capitals_selectivity": float(sel[ci, l, n]),
                "top_input_tokens": [tok.decode([t]) for t in topt[l, n].tolist()],
                "domain_profile": {d: float(prof[i, l, n]) for i, d in enumerate(dom_names)}}
        if extra:
            item.update(extra)
        if l >= 14:
            w = model.model.layers[l].mlp.down_proj.weight.detach()[:, n] * norm_w
            logit = W_U @ w
            item["promotes"] = [tok.decode([t]) for t in logit.topk(8).indices.tolist()]
            item["suppresses"] = [tok.decode([t]) for t in (-logit).topk(5).indices.tolist()]
        return item

    countries = [c for c, _ in cw.CAPITALS[:64]]
    ent_layers = [int(x) for x in args.entity_layers.split(",")]
    report["entity_identity_clean"] = entity_probe(model, H, tok, None, countries, ent_layers)
    log("[entity] clean " + " ".join(f"{k}={v:.2f}" for k, v in report["entity_identity_clean"].items()))
    report["cores"] = {}
    g = torch.Generator().manual_seed(args.seed + 99)
    for crow, core in picks:
        tagc = f"core{crow['lam']}"
        Zb = torch.full((L, N), -0.5, requires_grad=True)   # box starts empty (gated off)
        boxes = []
        for lam in [float(x) for x in args.box_lambdas.split(",")]:
            Zb = train_mask(model, H, lambda s: None, Zb, core, lam, args.box_steps, args.lr, box_loss, log,
                            f"box@{tagc}", straight_through=True)
            B = (Zb.detach() > 0).float() * (1 - core)
            row = {"lam": lam, "box_neurons": int(B.sum()), "box_frac_of_all": float(B.mean()),
                   "box_per_layer": B.sum(1).tolist(), "core_plus_box": held_scores(core + B)}
            boxes.append((row, B.clone()))
            log(f"[box@{tagc}] lam={lam} box={row['box_neurons']} ({row['box_frac_of_all']:.4f}) held-out cap acc="
                f"{row['core_plus_box']['capitals_heldout']['acc_masked']:.2f} (all {row['core_plus_box']['capitals_all']['acc_masked']:.2f}) "
                f"English d_CE={row['core_plus_box']['mean_d_ce']:.3f}")
        # choose on the fit capitals; held-out is only reported (Codex review 2026-11-12)
        box = max(boxes, key=lambda x: x[0]["core_plus_box"].get("capitals_fit", x[0]["core_plus_box"]["capitals_all"])["acc_masked"]
                  - 1e-5 * x[0]["box_neurons"])[1]
        groups = {"core": core.bool(), "dropped": (1 - core - box).bool(), "box": box.bool()}
        gstats, inside, examples = {}, {}, {}
        for name, G in groups.items():
            n = int(G.sum())
            if n == 0:
                continue
            rand = torch.zeros(L * N, dtype=torch.bool)
            rand[torch.randperm(L * N, generator=g)[:n]] = True
            rand = rand.view(L, N)
            gstats[name] = {"neurons": n,
                            "domain_selectivity_mean": {d: float(sel[i][G].mean()) for i, d in enumerate(dom_names)},
                            "random_same_size": {d: float(sel[i][rand].mean()) for i, d in enumerate(dom_names)},
                            "share_capitals_selective_gt2": float((sel[ci][G] > 2).float().mean()),
                            "random_capitals_selective_gt2": float((sel[ci][rand] > 2).float().mean())}
            inside[name] = {"top_token_categories": cat_share(G), "random_same_size": cat_share(rand),
                            "domain_variation_cv": float(cv[G].mean()), "random_cv": float(cv[rand].mean()),
                            "per_layer_neurons": G.sum(1).tolist()}
            cand = G.nonzero()
            top = cand[sel[ci][G].topk(min(12, n)).indices]
            examples[name] = [describe(l, nn) for l, nn in top.tolist()]
            log(f"[sig@{tagc}] {name}: n={n} sel " +
                " ".join(f"{d[:4]}={gstats[name]['domain_selectivity_mean'][d]:.2f}/{gstats[name]['random_same_size'][d]:.2f}" for d in dom_names) +
                f" | cap>2 {gstats[name]['share_capitals_selective_gt2']:.3f}/{gstats[name]['random_capitals_selective_gt2']:.3f} | " +
                " ".join(f"{c}={inside[name]['top_token_categories'][c]:.2f}/{inside[name]['random_same_size'][c]:.2f}" for c in cats) +
                f" | CV {inside[name]['domain_variation_cv']:.2f}/{inside[name]['random_cv']:.2f}")
        core_examples = {}
        for band, (lo, hi) in {"early": (0, 7), "mid": (7, 18), "late": (18, L)}.items():
            # most domain-general core neurons with the largest typical deviation (the core's workhorses)
            score = torch.where(core.bool(), prof.mean(0) / (1 + cv), torch.full_like(cv, float("-inf")))[lo:hi]
            core_examples[band] = [describe(lo + f // N, f % N) for f in score.reshape(-1).topk(10).indices.tolist()]
        rnd_core = torch.zeros(L, N)
        gr = torch.Generator().manual_seed(args.seed + 5)
        for l in range(L):
            k = int(core[l].sum())
            if k:
                rnd_core[l, torch.randperm(N, generator=gr)[:k]] = 1.0
        ent = {"core": entity_probe(model, H, tok, core, countries, ent_layers),
               "core_plus_box": entity_probe(model, H, tok, core + box, countries, ent_layers),
               "random_core": entity_probe(model, H, tok, rnd_core, countries, ent_layers)}
        for nm, e in ent.items():
            log(f"[entity@{tagc}] {nm} " + " ".join(f"{k}={v:.2f}" for k, v in e.items()))
        report["cores"][str(crow["lam"])] = {"core": crow, "box_sweep": [r for r, _ in boxes],
                                             "entity_identity": ent,
                                             "groups": gstats, "inside": inside,
                                             "most_capitals_selective": examples,
                                             "core_workhorses": core_examples}
        torch.save({"core": core, "box": box, "mean": mean}, out / f"masks_{tagc}.pt")
    report["signature_method"] = ("hidden-state version of expert-token-signatures: mean |act-mean| per domain, "
                                  "relative to the cross-domain mean; top tokens by |act-mean|; token categories; "
                                  "late-layer write direction through final norm + unembedding; random same-size "
                                  "neuron sets as the control")
    report["wall_s"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = round(cw.rss_gb(), 2)
    tmp = out / "english_core.json.tmp"
    tmp.write_text(json.dumps(report, indent=1))
    tmp.rename(out / "english_core.json")
    log(f"wrote {out/'english_core.json'} in {report['wall_s']}s, peak RSS {report['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
