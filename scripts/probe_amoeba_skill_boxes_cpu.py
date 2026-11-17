#!/usr/bin/env python3
"""Amoeba M0b-2: several skill boxes on one English core, with per-token gates.

CPU only (torch, fp32). Uses a core saved by probe_amoeba_english_core_cpu.py (zero ablation, so
everything here can later be exported). For each skill -- capitals, arithmetic, English->French
words -- a box is a straight-through mask over the non-core MLP neurons, learned on a fit split
and scored on a held-out split, by teacher-forced exact match of the whole answer:

    loss = mean KL(full model || core + box) over the answer positions + lam * box size

Then every box gets a per-token gate: g_l = sigmoid(x_l . w_l + b) from the MLP input at each
layer, so a box neuron at layer l contributes g_l * (act - mean). Gates start open (equal to
the ungated boxes) and are trained on a mix of every skill's fit split, a multi-turn
number-recall task and general English, with KL to the full model everywhere and a small
penalty on openness over general English. The question is whether gating removes the
interference seen in the A/B (an always-on capitals box broke multi-turn recall), while
keeping each skill.

Eval matrix (held-out): full model, core, core + each box, core + all boxes ungated,
core + all boxes gated; every skill, multi-turn recall, and English d_CE on held-out text.
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_amoeba_cone_width_cpu as cw  # noqa: E402
import probe_amoeba_english_core_cpu as ec  # noqa: E402

FRENCH = [("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"), ("book", "livre"),
          ("bread", "pain"), ("cheese", "fromage"), ("apple", "pomme"), ("tree", "arbre"), ("sun", "soleil"),
          ("moon", "lune"), ("star", "étoile"), ("sea", "mer"), ("river", "rivière"), ("mountain", "montagne"),
          ("city", "ville"), ("street", "rue"), ("car", "voiture"), ("train", "train"), ("boat", "bateau"),
          ("bird", "oiseau"), ("fish", "poisson"), ("horse", "cheval"), ("cow", "vache"), ("mouse", "souris"),
          ("flower", "fleur"), ("garden", "jardin"), ("door", "porte"), ("window", "fenêtre"), ("table", "table"),
          ("chair", "chaise"), ("bed", "lit"), ("kitchen", "cuisine"), ("friend", "ami"), ("mother", "mère"),
          ("father", "père"), ("brother", "frère"), ("sister", "sœur"), ("child", "enfant"), ("man", "homme"),
          ("woman", "femme"), ("king", "roi"), ("queen", "reine"), ("school", "école"), ("teacher", "professeur"),
          ("money", "argent"), ("time", "temps"), ("day", "jour"), ("night", "nuit"), ("week", "semaine"),
          ("year", "année"), ("morning", "matin"), ("evening", "soir"), ("winter", "hiver"), ("summer", "été"),
          ("spring", "printemps"), ("snow", "neige"), ("rain", "pluie"), ("wind", "vent"), ("fire", "feu"),
          ("milk", "lait"), ("egg", "œuf"), ("meat", "viande"), ("wine", "vin"), ("beer", "bière"),
          ("red", "rouge"), ("blue", "bleu"), ("green", "vert"), ("black", "noir"), ("white", "blanc"),
          ("yellow", "jaune"), ("big", "grand"), ("small", "petit"), ("new", "nouveau"), ("old", "vieux"),
          ("good", "bon"), ("bad", "mauvais"), ("happy", "heureux"), ("sad", "triste"), ("hot", "chaud"),
          ("cold", "froid"), ("hand", "main"), ("head", "tête"), ("eye", "œil"), ("heart", "cœur"),
          ("road", "route"), ("bridge", "pont"), ("church", "église"), ("market", "marché"), ("shop", "magasin"),
          ("letter", "lettre"), ("paper", "papier"), ("pen", "stylo"), ("song", "chanson"), ("game", "jeu"),
          ("love", "amour"), ("war", "guerre"), ("peace", "paix"), ("country", "pays"), ("island", "île")]
FR_SHOTS = "English: shoe\nFrench: chaussure\nEnglish: hat\nFrench: chapeau\n"


def tf_items(tok, prompts_answers):
    """Teacher-forced items: ids = prompt + answer, positions = indices predicting answer tokens."""
    items = []
    for p, a, label in prompts_answers:
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(a, add_special_tokens=False)["input_ids"]
        items.append({"ids": pi + ai, "pos": list(range(len(pi) - 1, len(pi) + len(ai) - 1)),
                      "tgt": ai, "label": label})
    return items


def build_tasks(tok, seed):
    rng = random.Random(seed)
    tasks = {}
    caps = [(cw.TEMPLATE.format(c), " " + k, c) for c, k in cw.CAPITALS]
    # full-word answers for multi-token capitals are taken as the first word in CAPITALS
    tasks["capitals"] = tf_items(tok, caps)
    pairs = set()
    while len(pairs) < 300:
        pairs.add((rng.randint(10, 99), rng.randint(10, 99)))
    tasks["arithmetic"] = tf_items(tok, [(f"{a} + {b} =", f" {a + b}", f"{a}+{b}") for a, b in sorted(pairs)])
    tasks["french"] = tf_items(tok, [(FR_SHOTS + f"English: {e}\nFrench:", f" {f}", e) for e, f in FRENCH])
    nums = rng.sample(range(10, 99), 80)
    fillers = ["What color is the sky? A: Blue.", "How many legs does a cat have? A: Four.",
               "What do bees make? A: Honey.", "Which season is coldest? A: Winter."]
    tasks["recall"] = tf_items(tok, [(f"Remember the number {n}. Q: {fillers[i % 4]} Q: What number did I ask you "
                                      f"to remember? A: The number is", f" {n}", str(n)) for i, n in enumerate(nums)])
    splits = {}
    for name, items in tasks.items():
        idx = list(range(len(items)))
        random.Random(seed + len(name)).shuffle(idx)
        h = len(idx) // 2
        splits[name] = {"fit": sorted(idx[:h]), "held": sorted(idx[h:])}
    return tasks, splits


class GatedBoxes:
    """MLP hook: used = mean + m * (act - mean), m = core + sum_b box_b * gate_b(x) (per token)."""

    def __init__(self, model, mean):
        self.mlps = [l.mlp for l in model.model.layers]
        self.mean = mean
        self.core, self.boxes, self.gates, self.active = None, {}, None, None
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(self.mlps)]

    def _hook(self, l):
        def fn(mod, args, out):
            if self.core is None:
                return None
            x = args[0]
            act = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            m = self.core[l]
            for name in self.active:
                b = self.boxes[name][l]
                if self.gates is not None and name in self.gates:
                    w, bias = self.gates[name]
                    g = torch.sigmoid((x * w[l]).sum(-1, keepdim=True) + bias[l])   # B,T,1
                    m = m + b * g
                else:
                    m = m + b
            used = self.mean[l] + m * (act - self.mean[l])
            used = torch.cat([act[:, :1], used[:, 1:]], 1)
            return mod.down_proj(used)
        return fn

    def set(self, core=None, active=(), gates=None):
        self.core, self.active, self.gates = core, list(active), gates


def groups_of(items, idx, cap=24):
    by = {}
    for i in idx:
        by.setdefault(len(items[i]["tgt"]), []).append(i)          # same answer length; prompts right-padded
    out = []
    for g in by.values():
        out += [g[j:j + cap] for j in range(0, len(g), cap)]
    return out


def run_logits(model, ids, pos=None):
    """Logits only where needed: a 152K-vocab head over every position dominates memory.

    pos: list (same positions for every row) or a [B, A] tensor (per-row positions).
    """
    h = model.model(input_ids=ids).last_hidden_state
    if pos is not None:
        if torch.is_tensor(pos) and pos.dim() == 2:
            h = h[torch.arange(h.shape[0])[:, None], pos]
        else:
            h = h[:, pos]
    return model.lm_head(h).float()


def batch_of(items, grp, pad=0):
    """Right-pad a group (causal model: padding after a row never changes its own logits)."""
    T = max(len(items[i]["ids"]) for i in grp)
    ids = torch.tensor([items[i]["ids"] + [pad] * (T - len(items[i]["ids"])) for i in grp])
    pos = torch.tensor([items[i]["pos"] for i in grp])
    return ids, pos


def task_kl_acc(model, H, items, idx, cfg, clean_cache, grad=False, prep=None):
    """Mean KL over answer positions vs the full model and teacher-forced exact-match accuracy.

    grad=True back-propagates each group's share of the mean KL immediately (gradient
    accumulation), so only one group's graph is alive at a time; prep() rebuilds any
    parameter-derived masks before each group.
    """
    n_total = sum(len(items[i]["pos"]) for i in idx)
    kl_tot, correct = 0.0, 0
    for grp in groups_of(items, idx):
        ids, pos = batch_of(items, grp)
        lc = clean_cache[(id(items), tuple(grp))]
        if prep is not None:
            prep()
        H.set(**cfg)
        if grad:
            lm = run_logits(model, ids, pos)
            kl = ec.kl_rows(lc, lm)
            (kl.sum() / n_total).backward()
        else:
            with torch.no_grad():
                lm = run_logits(model, ids, pos)
                kl = ec.kl_rows(lc, lm)
        H.set()
        kl_tot += float(kl.detach().sum())
        tgt = torch.tensor([items[i]["tgt"] for i in grp])
        correct += int((lm.detach().argmax(-1) == tgt).all(-1).sum())
    return kl_tot / max(1, n_total), correct / len(idx)


def cache_clean(model, H, items, idx, cache):
    for grp in groups_of(items, idx):
        key = (id(items), tuple(grp))
        if key in cache:
            continue
        ids, pos = batch_of(items, grp)
        H.set()
        with torch.no_grad():
            cache[key] = run_logits(model, ids, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--core-masks", required=True, help="masks_core<lam>.pt (zero ablation preferred)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skills", default="capitals,arithmetic,french")
    ap.add_argument("--box-lambdas", default="1,4,16")
    ap.add_argument("--box-steps", type=int, default=150)
    ap.add_argument("--gate-steps", type=int, default=200)
    ap.add_argument("--gate-penalty", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--gate-lr", type=float, default=0.01)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--n-english", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
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
    d = model.config.hidden_size
    blob = torch.load(args.core_masks)
    core, mean = blob["core"], blob["mean"]
    H = GatedBoxes(model, mean)
    t0 = time.time()
    rep = {"checkpoint": f"{args.model}@{os.path.basename(snap)[:8]}",
           "weights_sha256_12": {os.path.basename(w): cw.sha(w) for w in sorted(glob.glob(os.path.join(snap, "*.safetensors")))},
           "mlx2_source_revision": cw.git_rev(), "core_masks": args.core_masks, "core_kept": float(core.mean()),
           "ablation": "zero" if float(mean.abs().max()) == 0 else "mean", "args": vars(args), "device": "cpu"}

    tasks, splits = build_tasks(tok, args.seed)
    rep["task_sizes"] = {k: {s: len(v) for s, v in splits[k].items()} for k in tasks}
    corpus = json.loads(Path(args.corpus).read_text())["seqs"]
    rng = random.Random(args.seed + 3)
    eng_held, eng_fit = [], []
    for dname in ec.DOMAINS:
        s = [q[:args.seq_len] for q in corpus[dname]]
        rng.shuffle(s)
        eng_held += s[:args.n_english // 3]
        eng_fit += s[args.n_english // 3: args.n_english // 3 + 8]
    cache = {}
    for name, items in tasks.items():
        cache_clean(model, H, items, splits[name]["fit"] + splits[name]["held"], cache)
        cache_clean(model, H, items, splits[name]["fit"], cache)
        cache_clean(model, H, items, splits[name]["held"], cache)

    def english_dce(cfg):
        ce_c = ce_m = n = 0.0
        for i in range(0, len(eng_held), 4):
            ids = torch.tensor(eng_held[i:i + 4])
            with torch.no_grad():
                H.set()
                lc = run_logits(model, ids, list(range(1, ids.shape[1] - 1)))
                H.set(**cfg)
                lm = run_logits(model, ids, list(range(1, ids.shape[1] - 1)))
                H.set()
            tgt = ids[:, 2:].flatten()
            ce_c += nn.functional.cross_entropy(lc.flatten(0, 1), tgt, reduction="sum").item()
            ce_m += nn.functional.cross_entropy(lm.flatten(0, 1), tgt, reduction="sum").item()
            n += tgt.numel()
        return (ce_m - ce_c) / n

    def evaluate(cfg, tag):
        r = {}
        for name, items in tasks.items():
            if cfg is None:
                with torch.no_grad():
                    _, acc = task_kl_acc(model, H, items, splits[name]["held"], {}, cache)
                r[name] = {"acc": acc, "kl": 0.0}
            else:
                with torch.no_grad():
                    kl, acc = task_kl_acc(model, H, items, splits[name]["held"], cfg, cache)
                r[name] = {"acc": acc, "kl": float(kl)}
        r["english_d_ce"] = 0.0 if cfg is None else english_dce(cfg)
        log(f"[eval] {tag:22s} " + " ".join(f"{k}={v['acc']:.2f}" for k, v in r.items() if k != "english_d_ce")
            + f" english_dCE={r['english_d_ce']:.3f}")
        return r

    rep["eval"] = {"full": evaluate(None, "full"), "core": evaluate({"core": core}, "core")}

    # ---- 1. one straight-through box per skill
    free = 1.0 - core
    boxes, box_rows = {}, {}
    for skill in args.skills.split(","):
        items, fit = tasks[skill], splits[skill]["fit"]
        Z = torch.full((L, N), -0.5, requires_grad=True)
        opt = torch.optim.Adam([Z], lr=args.lr)
        rows = []
        for lam in [float(x) for x in args.box_lambdas.split(",")]:
            for step in range(args.box_steps):
                opt.zero_grad()

                def prep():
                    soft = torch.sigmoid(Z)
                    H.boxes = {"_train": free * ((Z > 0).float() + soft - soft.detach())}
                task_kl_acc(model, H, items, fit, {"core": core, "active": ["_train"]}, cache, grad=True, prep=prep)
                (lam * (free * torch.sigmoid(Z)).sum() / free.sum()).backward()
                opt.step()
            B = (Z.detach() > 0).float() * free
            H.boxes = {"_eval": B}
            with torch.no_grad():
                _, acc_h = task_kl_acc(model, H, items, splits[skill]["held"], {"core": core, "active": ["_eval"]}, cache)
                _, acc_f = task_kl_acc(model, H, items, fit, {"core": core, "active": ["_eval"]}, cache)
            rows.append({"lam": lam, "neurons": int(B.sum()), "held_acc": acc_h, "fit_acc": acc_f, "box": B})
            log(f"[box {skill}] lam={lam} neurons={int(B.sum())} held={acc_h:.2f} fit={acc_f:.2f}")
        # keep the smallest box within 0.05 of the best held-out accuracy
        # choose on the fit split; the held-out split is only reported (Codex review 2026-11-12)
        best = max(r["fit_acc"] for r in rows)
        pick = min([r for r in rows if r["fit_acc"] >= best - 0.05], key=lambda r: r["neurons"])
        boxes[skill] = pick["box"]
        box_rows[skill] = [{k: v for k, v in r.items() if k != "box"} for r in rows]
        box_rows[skill + "_picked"] = {"lam": pick["lam"], "neurons": pick["neurons"]}
    H.boxes = dict(boxes)
    rep["boxes"] = box_rows
    overlap = {}
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}&{b}"] = int((boxes[a].bool() & boxes[b].bool()).sum())
    rep["box_overlap_neurons"] = overlap
    for skill in names:
        rep["eval"][f"core+{skill}"] = evaluate({"core": core, "active": [skill]}, f"core+{skill}")
    rep["eval"]["core+all"] = evaluate({"core": core, "active": names}, "core+all (ungated)")

    # ---- 2. per-token gates for every box, trained jointly
    gates = {n: (torch.zeros(L, d, requires_grad=True), torch.full((L,), 2.0, requires_grad=True)) for n in names}
    opt = torch.optim.Adam([p for wb in gates.values() for p in wb], lr=args.gate_lr)
    mix = [(n, tasks[n], splits[n]["fit"]) for n in names + ["recall"]]
    eng_pos = list(range(1, args.seq_len, 4))                  # every 4th position keeps the head small
    eng_ids = [torch.tensor(eng_fit[i:i + 4]) for i in range(0, len(eng_fit), 4)]
    eng_clean = []
    for ids in eng_ids:
        with torch.no_grad():
            H.set()
            eng_clean.append(run_logits(model, ids, eng_pos))
    for step in range(args.gate_steps):
        cfg = {"core": core, "active": names, "gates": gates}
        opt.zero_grad()
        loss = 0.0
        for n, items, fit in mix:
            kl, _ = task_kl_acc(model, H, items, fit, cfg, cache, grad=True)
            loss += kl
        ids, lc = eng_ids[step % len(eng_ids)], eng_clean[step % len(eng_ids)]
        H.set(**cfg)
        lm = run_logits(model, ids, eng_pos)
        H.set()
        ekl = ec.kl_rows(lc, lm).mean()
        ekl.backward()
        loss += float(ekl.detach())
        # openness penalty on general English: prefer closed gates where boxes are not needed
        with torch.enable_grad():
            openness = 0.0
            caps_x = {}

            def grab(l):
                def f(mod, a, o):
                    caps_x[l] = a[0]
                return f
            hs = [model.model.layers[l].mlp.register_forward_hook(grab(l)) for l in range(L)]
            H.set(**cfg)
            with torch.no_grad():
                model.model(input_ids=ids)
            H.set()
            for h in hs:
                h.remove()
            for n, (w, bias) in gates.items():
                for l in range(L):
                    if boxes[n][l].any():
                        openness = openness + torch.sigmoid((caps_x[l].detach() * w[l]).sum(-1) + bias[l]).mean()
        if torch.is_tensor(openness):
            (args.gate_penalty * openness).backward()
        opt.step()
        if step % 25 == 0 or step == args.gate_steps - 1:
            log(f"{time.strftime('%H:%M:%S')}   gates step {step} loss={loss:.4f} openness={float(openness.detach()) if torch.is_tensor(openness) else 0.0:.3f}")
    rep["eval"]["core+all_gated"] = evaluate({"core": core, "active": names, "gates": gates}, "core+all (gated)")
    torch.save({"core": core, "mean": mean, "boxes": boxes,
                "gates": {n: (w.detach(), b.detach()) for n, (w, b) in gates.items()}}, out / "skill_boxes.pt")
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["peak_rss_gb"] = round(cw.rss_gb(), 2)
    tmp = out / "skill_boxes.json.tmp"
    tmp.write_text(json.dumps(rep, indent=1))
    tmp.rename(out / "skill_boxes.json")
    log(f"wrote {out/'skill_boxes.json'} in {rep['wall_s']}s, peak RSS {rep['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
