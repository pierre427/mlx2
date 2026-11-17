#!/usr/bin/env python3
"""Amoeba M0d: box-ify without topics, then back-label the boxes.

CPU only (torch, fp32). Starts from an English core (masks_core<lam>.pt, mean ablation). The
"knowledge pool" is every MLP neuron outside the core. Nothing below uses a topic label to
form boxes:

1. Stream an unlabelled mixed corpus: the M0a-3 corpus (TinyStories, repo markdown, donor
   encyclopedic text) plus the short task prompts (capital phrasings, sums, English->French,
   number recall) as plain text, the way a broad corpus would contain such material.
   Per pool neuron and sequence, record its peak |act - mean| over positions >= 1.
2. Cluster pool neurons by *sequence-level* co-firing (spherical k-means on per-neuron
   sequence profiles). Sequence level keeps a box's early neurons (at the entity token) and
   late neurons (at the answer position) together.
3. Back-label each cluster:
     descriptive : per-source activation profile, top sequences (decoded), layer histogram
     functional  : ablate the cluster from the full model (specificity: which domains and
                   tasks get worse) and add it back to the core (sufficiency: which tasks
                   recover). Task accuracy is teacher-forced exact match on held-out halves.
4. Agreement with supervised boxes (M0b-2 skill_boxes.pt): share of each supervised box that
   falls in its best cluster, and the enrichment over chance.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amoeba_device  # noqa: E402
import probe_amoeba_cone_width_cpu as cw  # noqa: E402
import probe_amoeba_english_core_cpu as ec  # noqa: E402
import probe_amoeba_skill_boxes_cpu as sb  # noqa: E402


def spherical_kmeans(X, k, iters, seed):
    """X: [n_items, dim] rows L2-normalised. Returns labels [n_items]."""
    g = amoeba_device.generator(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].sum(0)
                C[j] = c / c.norm().clamp_min(1e-8)
            else:                                   # re-seed an empty cluster
                C[j] = X[torch.randint(0, X.shape[0], (1,), generator=g)].squeeze(0)
    return (X @ C.T).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--core-masks", required=True, help="masks_core<lam>.pt (mean ablation)")
    ap.add_argument("--skill-boxes", default="", help="skill_boxes.pt from M0b-2, for agreement only")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-corpus-seqs", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--n-english-eval", type=int, default=12)
    ap.add_argument("--device", default="cpu", help="cpu or mps (Apple GPU; run inside an owned GPU window)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    amoeba_device.setup(args.device)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    logf = open(out / "run.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = snapshot_download(args.model, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval().to(amoeba_device.place())
    for p in model.parameters():
        p.requires_grad_(False)
    L, N = model.config.num_hidden_layers, model.config.intermediate_size
    blob = torch.load(args.core_masks)
    core, mean = blob["core"], blob["mean"]
    pool = (core == 0)
    pool_idx = pool.nonzero()                                   # [P, 2] (layer, neuron)
    P = pool_idx.shape[0]
    t0 = time.time()
    rep = {"checkpoint": f"{args.model}@{os.path.basename(snap)[:8]}", "mlx2_source_revision": cw.git_rev(),
           "core_masks": args.core_masks, "pool_neurons": P, "args": vars(args), "device": amoeba_device.DEVICE,
           "labels_used_to_form_boxes": "none (sources and tasks only used for back-labelling)"}
    log(f"knowledge pool: {P} neurons outside the core ({float(core.mean()):.3f} kept)")

    # ---- 1. unlabelled mixed corpus
    corpus = json.loads(Path(args.corpus).read_text())["seqs"]
    rng = random.Random(args.seed)
    seqs, source = [], []
    per = args.n_corpus_seqs // len(ec.DOMAINS)
    for dn in ec.DOMAINS:
        s = [q[:args.seq_len] for q in corpus[dn]]
        rng.shuffle(s)
        seqs += s[:per]
        source += [dn] * per
    tasks, splits = sb.build_tasks(tok, args.seed)
    for name, items in tasks.items():
        for it in items:
            seqs.append(it["ids"])
            source.append(name)
    cap5 = []
    for tpl in ["{}'s capital city is", "The capital city of {} is", "Q: What is the capital of {}? A:"]:
        for c, k in cw.CAPITALS:
            cap5.append(tok(tpl.format(c) + " " + k, add_special_tokens=False)["input_ids"])
    seqs += cap5
    source += ["capitals"] * len(cap5)
    order = list(range(len(seqs)))
    rng.shuffle(order)                                          # no source ordering survives
    seqs, source = [seqs[i] for i in order], [source[i] for i in order]
    S = len(seqs)
    log(f"corpus: {S} sequences from {sorted(set(source))}")

    # ---- per pool neuron x sequence: peak |act - mean|
    feat = torch.zeros(S, P)
    layer_cols = {l: (pool_idx[:, 0] == l).nonzero().flatten() for l in range(L)}
    layer_neu = {l: pool_idx[layer_cols[l], 1] for l in range(L)}
    state = {}

    def grab(l):
        def fn(mod, a, o):
            x = a[0]
            act = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            dev = (act[:, 1:, layer_neu[l]] - mean[l, layer_neu[l]]).abs()
            m = state["mask"][:, 1:, None]
            state["peak"][l] = (dev * m).amax(1)                # B x n_l
        return fn
    hooks = [model.model.layers[l].mlp.register_forward_hook(grab(l)) for l in range(L)]
    by_len = {}
    for i, q in enumerate(seqs):
        by_len.setdefault(len(q), []).append(i)
    batches = []
    for ln in sorted(by_len):
        g = by_len[ln]
        batches += [g[j:j + 16] for j in range(0, len(g), 16)]
    for bi, grp in enumerate(batches):
        ids = torch.tensor([seqs[i] for i in grp])
        state["mask"] = torch.ones(ids.shape)
        state["peak"] = {}
        with torch.no_grad():
            model.model(input_ids=ids)
        for l in range(L):
            if len(layer_cols[l]):
                feat[torch.tensor(grp)[:, None], layer_cols[l][None, :]] = state["peak"][l]
        if bi % 100 == 0:
            log(f"{time.strftime('%H:%M:%S')} streamed {bi}/{len(batches)} batches")
    for h in hooks:
        h.remove()

    # ---- 2. cluster by sequence-level co-firing
    Z = feat / feat.mean(0, keepdim=True).clamp_min(1e-8)       # per-neuron relative profile
    Z = Z - Z.mean(0, keepdim=True)
    X = Z.T / Z.T.norm(dim=1, keepdim=True).clamp_min(1e-8)     # P x S
    lab = spherical_kmeans(X, args.k, args.iters, args.seed)
    sizes = torch.bincount(lab, minlength=args.k)
    log(f"clusters: sizes min {int(sizes.min())} median {int(sizes.median())} max {int(sizes.max())}")

    # ---- 3. back-label
    srcs = sorted(set(source))
    src_t = {s: torch.tensor([i for i, x in enumerate(source) if x == s]) for s in srcs}
    H = sb.GatedBoxes(model, mean)
    cache = {}
    eval_sets = {n: splits[n]["held"] for n in tasks}
    for n, items in tasks.items():
        sb.cache_clean(model, H, items, eval_sets[n], cache)
    eng = []
    for dn in ec.DOMAINS:
        s = [q[:args.seq_len] for q in corpus[dn][-40:]]
        eng += s[:args.n_english_eval // 3]

    def score(cfg):
        r = {}
        for n, items in tasks.items():
            _, acc = sb.task_kl_acc(model, H, items, eval_sets[n], cfg, cache)
            r[n] = acc
        ce = [0.0, 0.0]
        cnt = 0
        for i in range(0, len(eng), 4):
            ids = torch.tensor(eng[i:i + 4])
            pos = list(range(1, ids.shape[1] - 1))
            tgt = ids[:, 2:].flatten()
            for j, c in enumerate(({}, cfg)):
                H.set(**c)
                with torch.no_grad():
                    lg = sb.run_logits(model, ids, pos)
                H.set()
                ce[j] += nn.functional.cross_entropy(lg.flatten(0, 1), tgt, reduction="sum").item()
            cnt += tgt.numel()
        r["english_d_ce"] = (ce[1] - ce[0]) / cnt
        return r

    full = score({})
    core_only = score({"core": core})
    rep["full"], rep["core_only"] = full, core_only
    log(f"full {full} | core {core_only}")
    clusters = []
    ones = torch.ones(L, N)
    for j in range(args.k):
        members = pool_idx[lab == j]
        cm = torch.zeros(L, N)
        if len(members):
            cm[members[:, 0], members[:, 1]] = 1.0
        prof = {s: float(Z[src_t[s]][:, lab == j].mean()) for s in srcs} if len(members) else {}
        act_seq = feat[:, lab == j].mean(1) if len(members) else torch.zeros(S)
        top = act_seq.topk(5).indices.tolist()
        H.boxes = {"c": cm}
        abl = score({"core": ones - cm})                         # remove the cluster from the full model
        add = score({"core": core, "active": ["c"]})             # add it to the core
        row = {"cluster": j, "neurons": len(members), "per_layer": cm.sum(1).tolist(), "source_profile": prof,
               "top_sequences": [{"source": source[i], "text": tok.decode(seqs[i])[:160]} for i in top],
               "ablate_from_full": {k: v - full[k] for k, v in abl.items()},
               "add_to_core": {k: v - core_only[k] for k, v in add.items()}}
        clusters.append(row)
        best_src = max(prof, key=prof.get) if prof else None
        log(f"[c{j:02d}] n={len(members)} top-source={best_src} | ablate: " +
            " ".join(f"{k[:5]}{v:+.2f}" for k, v in row["ablate_from_full"].items()) + " | add: " +
            " ".join(f"{k[:5]}{v:+.2f}" for k, v in row["add_to_core"].items()))
    rep["clusters"] = clusters

    # ---- 4. agreement with supervised boxes
    if args.skill_boxes:
        skb = torch.load(args.skill_boxes)["boxes"]
        agree = {}
        for name, B in skb.items():
            inpool = B.bool() & pool
            n_b = int(inpool.sum())
            if not n_b:
                continue
            labels_of_box = lab[(inpool[pool_idx[:, 0], pool_idx[:, 1]])]
            counts = torch.bincount(labels_of_box, minlength=args.k)
            jbest = int(counts.argmax())
            share = float(counts[jbest]) / n_b
            chance = float(sizes[jbest]) / P
            agree[name] = {"box_neurons_in_pool": n_b, "best_cluster": jbest, "share_in_best": share,
                           "cluster_size": int(sizes[jbest]), "chance_share": chance,
                           "enrichment": share / max(chance, 1e-9),
                           "top3": [(int(i), int(counts[i])) for i in counts.topk(3).indices]}
            log(f"[agree] {name}: {share:.2f} of its {n_b} pool neurons in c{jbest:02d} "
                f"(size {int(sizes[jbest])}, chance {chance:.3f}, x{share / max(chance, 1e-9):.1f})")
        rep["supervised_agreement"] = agree
    torch.save({"labels": lab, "pool_idx": pool_idx}, out / "clusters.pt")
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["peak_rss_gb"] = round(cw.rss_gb(), 2)
    tmp = out / "unsupervised_boxes.json.tmp"
    tmp.write_text(json.dumps(rep, indent=1))
    tmp.rename(out / "unsupervised_boxes.json")
    log(f"wrote {out/'unsupervised_boxes.json'} in {rep['wall_s']}s, peak RSS {rep['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
