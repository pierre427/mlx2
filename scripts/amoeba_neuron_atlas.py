#!/usr/bin/env python3
"""Neuron atlas for a dense donor: stable ids, virtual experts and tags for every MLP neuron.

CPU or MPS, forward-only. Dense models have no expert ids, so (layer, neuron) is only an
address. The atlas gives every neuron:
  global id       : "<artifact>:L<layer>:N<neuron>" (artifact = model@snapshot[:8])
  virtual expert  : within each layer, neurons clustered by sequence-level co-firing on a
                    mixed corpus (spherical k-means, --experts-per-layer); id "L<l>:VE<k>".
                    The dense analogue of an MoE expert id; marked "virtual" wherever used.
  tags            : domain profile (mean |act - mean| per corpus domain, relative), top
                    tokens by |act - mean|, and the domain it is most selective for
  memberships     : which registered boxes/cores (docs/research/amoeba-box-registry) hold it
Outputs: atlas.pt (tensors), atlas_summary.json (per virtual expert: size, top tokens, domain
profile, box overlap) and neuron_tags.jsonl (one compact line per neuron).
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
import amoeba_device  # noqa: E402
import probe_amoeba_cone_width_cpu as cw  # noqa: E402
import probe_amoeba_skill_boxes_cpu as sb  # noqa: E402
import probe_amoeba_unsupervised_boxes_cpu as ub  # noqa: E402

DOMAINS = ["tinystories_train", "repo_markdown", "donor_generated", "tasks"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--registry", default="docs/research/amoeba-box-registry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=2400)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--experts-per-layer", type=int, default=32)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--top-tokens", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu or mps (Apple GPU; run inside an owned GPU window)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    amoeba_device.setup(args.device)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = snapshot_download(args.model, local_files_only=True)
    artifact = f"{args.model}@{os.path.basename(snap)[:8]}"
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval().to(amoeba_device.place())
    for p in model.parameters():
        p.requires_grad_(False)
    L, N = model.config.num_hidden_layers, model.config.intermediate_size
    t0 = time.time()

    corpus = json.loads(Path(args.corpus).read_text())["seqs"]
    rng = random.Random(args.seed)
    seqs, src = [], []
    per = args.n_seqs // 3
    for dn in DOMAINS[:3]:
        s = [q[:args.seq_len] for q in corpus[dn]]
        rng.shuffle(s)
        seqs += s[:per]
        src += [dn] * per
    tasks, _ = sb.build_tasks(tok, args.seed)
    for items in tasks.values():
        for it in items:
            seqs.append(it["ids"])
            src.append("tasks")
    S = len(seqs)

    # running mean of activations (first pass, sample) for |act - mean|
    msum = torch.zeros(L, N)
    cnt = [0]
    peak = torch.zeros(S, L, N, dtype=torch.float16)
    dsum = torch.zeros(len(DOMAINS), L, N)
    dcnt = torch.zeros(len(DOMAINS))
    topv = torch.full((L, N, args.top_tokens), -1.0)
    topt = torch.zeros((L, N, args.top_tokens), dtype=torch.long)
    st = {"pass": 1}

    def hook(l):
        def fn(mod, a, o):
            x = a[0]
            act = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            body = act[:, 1:]
            if st["pass"] == 1:
                msum[l] += body.reshape(-1, N).sum(0).cpu()
                if l == 0:
                    cnt[0] += body.shape[0] * body.shape[1]
                return None
            dev = (body - st["mean"][l]).abs()
            peak[st["rows"], l] = dev.amax(1).half().cpu()
            di = st["dom"]
            dsum[di, l] += dev.reshape(-1, N).sum(0).cpu()
            if l == 0:
                dcnt[di] += dev.shape[0] * dev.shape[1]
            flat = dev.reshape(-1, N)
            tk = st["tok"].reshape(-1)
            v, i = flat.topk(min(args.top_tokens, flat.shape[0]), dim=0)
            cv = torch.cat([topv[l], v.T.cpu()], 1)
            ct = torch.cat([topt[l], tk[i].T.cpu()], 1)
            sv, si = cv.topk(args.top_tokens, dim=1)
            topv[l], topt[l] = sv, ct.gather(1, si)
            return None
        return fn
    hs = [model.model.layers[l].mlp.register_forward_hook(hook(l)) for l in range(L)]
    by = {}
    for i, q in enumerate(seqs):
        by.setdefault((src[i], len(q)), []).append(i)
    batches = []
    for (d, ln), g in by.items():
        batches += [(d, g[j:j + 16]) for j in range(0, len(g), 16)]
    with torch.no_grad():
        for d, g in batches[::6]:                                   # mean from a 1/6 sample
            model.model(input_ids=torch.tensor([seqs[i] for i in g]))
        st["pass"], st["mean"] = 2, (msum / max(1, cnt[0])).to(amoeba_device.DEVICE)
        for bi, (d, g) in enumerate(batches):
            ids = torch.tensor([seqs[i] for i in g])
            st["rows"], st["dom"], st["tok"] = torch.tensor(g), DOMAINS.index(d), ids[:, 1:]
            model.model(input_ids=ids)
            if bi % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} {bi}/{len(batches)} batches", flush=True)
    for h in hs:
        h.remove()
    prof = dsum / dcnt[:, None, None].clamp_min(1)
    sel = prof / prof.mean(0, keepdim=True).clamp_min(1e-8)

    # virtual experts per layer: co-firing over sequences
    ve = torch.zeros(L, N, dtype=torch.long)
    for l in range(L):
        F = peak[:, l].float()
        Z = F / F.mean(0, keepdim=True).clamp_min(1e-8)
        Z = Z - Z.mean(0, keepdim=True)
        X = Z.T / Z.T.norm(dim=1, keepdim=True).clamp_min(1e-8)
        ve[l] = ub.spherical_kmeans(X, args.experts_per_layer, args.iters, args.seed + l)
    print(f"{time.strftime('%H:%M:%S')} virtual experts done", flush=True)

    # memberships from the registry
    member = {}
    for f in sorted(glob.glob(str(Path(args.registry) / "*.json"))):
        rec = json.loads(Path(f).read_text())
        if rec["origin"]["donor"]["artifact"] != artifact:
            continue
        for u in rec["origin"]["units"]:
            for n in u["neurons"]:
                member.setdefault((u["layer"], n), []).append(rec["box_id"])

    dec = {}

    def tokstr(t):
        if t not in dec:
            dec[t] = tok.decode([t])
        return dec[t]
    with open(out / "neuron_tags.jsonl", "w") as fh:
        for l in range(L):
            for n in range(N):
                top_dom = int(sel[:, l, n].argmax())
                fh.write(json.dumps({"id": f"{artifact}:L{l}:N{n}", "layer": l, "neuron": n,
                                     "virtual_expert": f"L{l}:VE{int(ve[l, n]):02d}",
                                     "top_tokens": [tokstr(t) for t in topt[l, n].tolist()[:5]],
                                     "most_selective_domain": DOMAINS[top_dom],
                                     "selectivity": round(float(sel[top_dom, l, n]), 3),
                                     "boxes": member.get((l, n), [])}, ensure_ascii=False) + "\n")
    summary = {"schema": "mlx2.amoeba-neuron-atlas.v1", "artifact": artifact, "layers": L, "neurons_per_layer": N,
               "experts_per_layer": args.experts_per_layer, "domains": DOMAINS, "n_sequences": S,
               "mlx2_source_revision": cw.git_rev(), "device": amoeba_device.DEVICE,
               "virtual_experts": {}}
    for l in range(L):
        for k in range(args.experts_per_layer):
            m = ve[l] == k
            if not m.any():
                continue
            idx = m.nonzero().flatten()
            dp = prof[:, l, idx].mean(1)
            boxes = {}
            for n in idx.tolist():
                for b in member.get((l, n), []):
                    boxes[b] = boxes.get(b, 0) + 1
            counts = {}
            for n in idx.tolist():
                for t in topt[l, n].tolist()[:3]:
                    counts[t] = counts.get(t, 0) + 1
            summary["virtual_experts"][f"L{l}:VE{k:02d}"] = {
                "neurons": int(m.sum()), "domain_profile": {d: round(float(v), 4) for d, v in zip(DOMAINS, dp)},
                "most_selective_domain": DOMAINS[int((dp / prof[:, l].mean(1).clamp_min(1e-8)).argmax())],
                "common_top_tokens": [tokstr(t) for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:8]],
                "box_overlap": boxes}
    torch.save({"virtual_expert": ve, "domain_profile": prof, "selectivity": sel, "top_tokens": topt,
                "artifact": artifact}, out / "atlas.pt")
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["peak_rss_gb"] = round(cw.rss_gb(), 2)
    (out / "atlas_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote atlas for {artifact}: {L * N} neurons, {len(summary['virtual_experts'])} virtual experts, "
          f"{summary['wall_s']}s, peak RSS {summary['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
