#!/usr/bin/env python3
"""FN part 2 inputs (FN_BOXES_PLAN section 1.B/C, FN_DAY_PLAN stage part2), CPU only, no model forward.

1. Atlas sequences -> source rows. The Night A atlas (`fn-nightA-*/atlas`) stores per-sequence expert peaks in
   capture order plus each sequence's domain, not its text. The order is reproduced exactly by re-running the carve
   driver's corpus split (research/amoeba/carve/corpus._jsonl, same seed and counts, the FN tokenizer, which is what
   Host.encode calls) with each row's index carried along; the per-domain sequence counts must equal the atlas
   meta's, or the script stops.
2. Clusters: spherical k-means (k = --k, default 48) of the POOL experts (outside T2) on their relative, centred
   sequence profiles: x[e] = peak[:, e] over the corpus sequences (task sequences excluded), centred per expert and
   scaled to unit norm (the 0.6B / 27B atlas normalisation). Deterministic (seeded, k-means++ init, best of 4).
3. Candidates: a cluster becomes a candidate box when it has >= --min-experts experts and its top --top-seqs sequences
   (by the cluster's mean normalised peak) are not >= 50% from one part-1 cell whose box passed picking; otherwise it
   is that cell's box already and is recorded as support. Each candidate's carve corpus is the source rows of its
   top sequences (deduplicated, full row text, never measurement sets), domain "cluster" (generations are added by
   the stage inner).
No task boxes (Pierre 2026-11-16: FN boxes come only from the atlas clusters and the corpus topic cells; capitals and
arithmetic stay functional ladder probes, never carved as boxes, never counted).
Writes OUT/clusters.json (assignments, sizes, profiles, verdicts), OUT/corpora/<box>.jsonl, OUT/boxes.json.

Usage: amoeba_fn_part2_build.py --nighta DIR --t2 NPZ --partition-json PART1_PARTITION.json --part1-boxes BOXES.json
       --model DIR --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def corpus_order(path, encode, seq_len, n_mean, n_fit, n_held, seed):
    """research/amoeba/carve/corpus._jsonl, with the row index carried next to each sequence."""
    rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r.get("domain", "text"), []).append((i, r))
    doms = sorted(by)
    split = {}
    rng = random.Random(seed)
    SPLITS = ("held", "mean", "fit", "spare")

    def bucket(source):
        h = hashlib.sha256(f"{seed}:{source}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64
    for dn in doms:
        out = {k: [] for k in SPLITS}
        free = []
        for i, r in by[dn]:
            ids = encode(r["text"])
            if len(ids) < seq_len:
                continue
            sp = r.get("split")
            if sp == "fresh":
                sp = "held"
            if sp in out:
                out[sp].append(i)
            else:
                free.append((bucket(str(r.get("source", r["text"][:64]))), i))
        free.sort(key=lambda t: t[0])
        km, kf = n_mean // len(doms), n_fit // len(doms)
        j = 0
        for name, k in (("held", n_held), ("mean", km), ("fit", kf)):
            take = max(0, k - len(out[name]))
            out[name] += [i for _, i in free[j:j + take]]
            j += take
        out["spare"] += [i for _, i in free[j:]]
        for k in SPLITS:
            rng.shuffle(out[k])
        split[dn] = out
    return rows, split, doms


def spherical_kmeans(X, k, seed, iters=50, restarts=4):
    best = None
    for rs in range(restarts):
        rng = np.random.default_rng(seed + rs)
        c = [X[rng.integers(len(X))]]
        d = np.full(len(X), np.inf)
        for _ in range(1, k):                          # k-means++ on cosine distance
            d = np.minimum(d, 1 - X @ c[-1])
            p = np.clip(d, 0, None) ** 2
            c.append(X[rng.choice(len(X), p=p / p.sum())])
        C = np.stack(c)
        for _ in range(iters):
            lab = (X @ C.T).argmax(1)
            Cn = np.stack([X[lab == j].sum(0) if (lab == j).any() else C[j] for j in range(k)])
            Cn /= np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-12
            if np.allclose(Cn, C):
                break
            C = Cn
        score = float((X * C[lab]).sum())
        if best is None or score > best[0]:
            best = (score, lab, C)
    return best[1], best[2], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nighta", required=True, help="Night A root (T2) and, unless --atlas-run, its atlas")
    ap.add_argument("--atlas-run", default="", help="a recaptured atlas carve run dir (carve.json + atlas/)")
    ap.add_argument("--t2", required=True)
    ap.add_argument("--partition-json", required=True, help="part-1 partition.json (verdicts: which cells passed)")
    ap.add_argument("--part1-boxes", required=True, help="part-1 boxes.json (box -> corpus cell)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--min-experts", type=int, default=8)
    ap.add_argument("--top-seqs", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-field", default="", help="cluster only sequences whose source row has this field == 'fit' "
                    "(probe atlas: probe_split); the other sequences are kept for the tagger's held confirmation")
    ap.add_argument("--tagging-out", action="store_true", help="also write items.json + pool_clusters.npz for "
                    "scripts/amoeba_topics_label.py (one item per atlas sequence, in peak_L row order)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "corpora").mkdir(parents=True, exist_ok=False)
    na = Path(a.atlas_run).parent if a.atlas_run else Path(a.nighta)
    ar = Path(a.atlas_run) if a.atlas_run else na / "atlas-run"
    carve = json.loads((ar / "carve.json").read_text())
    atlas_dir = Path(carve.get("args", {}).get("atlas_dir") or (ar / "atlas"))
    args = carve["args"]
    meta = json.loads((atlas_dir / "meta.json").read_text())
    cpath = args["corpus"].removeprefix("jsonl:")
    want = {k: v for k, v in meta["seq_sources"].items() if not k.startswith("task_")}
    if meta.get("seq_docs"):
        # windowed-loader atlas (recaptured 2026-11-16): the capture recorded every sequence's source document
        rows = [json.loads(x) for x in Path(cpath).read_text().split("\n") if x.strip()]
        from research.amoeba.carve.corpus import _doc_id
        by_doc = {}
        for i, r in enumerate(rows):
            by_doc.setdefault(r.get("doc_id") or _doc_id(r["text"]), i)
        src = np.load(atlas_dir / "atlas_stats.npz")["seq_source"]
        held_docs = {k for k, i in by_doc.items() if rows[i].get("split") in ("held", "fresh")}
        order, ambiguous = [], 0
        for cl, dn in zip(meta["seq_docs"], src):
            if str(dn).startswith("task_"):
                continue
            if not cl:
                raise SystemExit("atlas seq_docs do not cover every corpus sequence")
            pick = next((c for c in cl if c[0] not in held_docs), None)     # never a held document
            if pick is None:
                raise SystemExit(f"atlas sequence maps only to held documents ({dn})")
            ambiguous += len({c[0] for c in cl}) > 1
            order.append((by_doc[pick[0]], str(dn)))
        if len(order) != sum(want.values()):
            raise SystemExit("atlas seq_docs do not cover every corpus sequence")
    else:
        ambiguous = 0
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        enc = lambda t: tok.encode(t, add_special_tokens=False)        # noqa: E731  (Host.encode)
        rows, split, doms = corpus_order(cpath, enc, args["seq_len"], args["n_mean"], args["n_fit"], args["n_held"],
                                         args["seed"])
        per = args["atlas_seqs"] // (len(doms) + 1)
        order = []
        for dn in doms:
            pool = split[dn]["fit"] + split[dn]["spare"]
            order += [(i, dn) for i in pool[:per]]
    got = Counter(dn for _, dn in order)
    if dict(got) != want:
        bad = {k: (got.get(k), want.get(k)) for k in set(got) | set(want) if got.get(k) != want.get(k)}
        raise SystemExit(f"atlas order not reproduced: per-domain counts differ {bad}")
    S = len(order)
    t2 = np.load(a.t2)["core"] > 0
    L, E = t2.shape
    pool = np.argwhere(~t2)                              # (layer, expert) of every pool expert
    X = np.zeros((len(pool), S), np.float32)
    for l in range(L):
        pk = np.load(atlas_dir / f"peak_L{l:02d}.npy", mmap_mode="r")[:S].astype(np.float32)   # [S, E]
        sel = pool[:, 0] == l
        X[sel] = pk[:, pool[sel, 1]].T
    fitsel = np.ones(S, bool)
    if a.fit_field:
        fitsel = np.array([rows[i].get(a.fit_field) == "fit" for i, _ in order])
    Xall = X
    X = X[:, fitsel]
    X -= X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    lab, C, score = spherical_kmeans(X, a.k, a.seed)
    fit_idx = np.flatnonzero(fitsel)
    from research.amoeba.carve.corpus import _doc_id, cut_at_first_code, shared_prefixes
    prefixes = {}                                        # per source domain, as the windowed loader strips them
    if args.get("strip_prefix"):
        bydom = {}
        for r in rows:
            t = r["text"]
            if args.get("code_cut") and str(r.get("domain", "")).startswith("code"):
                t = cut_at_first_code(t)
            bydom.setdefault(r.get("domain", "text"), []).append(t)
        prefixes = {d: (shared_prefixes(ts, args["strip_prefix"]) or ("",))[0] for d, ts in bydom.items()}
    part1 = json.loads(Path(a.partition_json).read_text())
    passed_cells = {}
    boxes1 = json.loads(Path(a.part1_boxes).read_text())
    cell_of = {b["box"]: b["cell"] for b in boxes1}
    for n in part1.get("passing", []):
        passed_cells[cell_of.get(n, n)] = n
    clusters, boxes = [], []
    for j in range(a.k):
        m = lab == j
        n = int(m.sum())
        if n == 0:
            continue
        prof = X[m].mean(0)
        top = fit_idx[np.argsort(-prof)[:a.top_seqs]]          # indices into the atlas order (fit sequences only)
        src = Counter(order[t][1] for t in top)
        dom, cnt = src.most_common(1)[0]
        rec = {"cluster": j, "experts": n, "layers": Counter(int(x) for x in pool[m, 0]).most_common(6),
               "top_sources": src.most_common(5), "dominant_share": cnt / len(top),
               "top_items": [{"seq": int(t), "probe_id": rows[order[t][0]].get("probe_id"),
                              "topics": rows[order[t][0]].get("topics")} for t in top]}
        if n < a.min_experts:
            rec["verdict"] = f"< {a.min_experts} experts"
        elif cnt / len(top) >= 0.5 and dom in passed_cells:
            rec["verdict"] = f"support of part-1 box {passed_cells[dom]} ({cnt}/{len(top)} top sequences from {dom})"
        else:
            name = f"cluster{j:02d}"
            seen, out_rows = set(), []
            for t in top:
                i = order[t][0]
                if i in seen:
                    continue
                seen.add(i)
                r = dict(rows[i])
                text = r["text"]
                od = order[t][1]
                if str(od).startswith("code"):            # review 2026-11-16: the box carves what the atlas saw
                    text = cut_at_first_code(text)
                pre = prefixes.get(od)
                if pre and text.startswith(pre):
                    text = text[len(pre):]
                out_rows.append({"text": text, "domain": "cluster", "source": r.get("source", ""),
                                 "doc_id": r.get("doc_id") or _doc_id(r["text"]), "atlas_domain": od,
                                 "atlas_seq": int(t), "probe_id": r.get("probe_id"), "topics": r.get("topics")})
            (out / "corpora" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
            rec["verdict"] = "candidate"
            boxes.append({"box": name, "cell": f"atlas cluster {j}", "corpus_rows": len(out_rows), "kind": "cluster",
                          "experts_in_cluster": n, "top_sources": src.most_common(3)})
        clusters.append(rec)
    (out / "clusters.json").write_text(json.dumps({
        "schema": "mlx2.amoeba-fn-part2.v1", "atlas_root": str(na), "fit_field": a.fit_field or None, "t2": a.t2, "k": a.k, "pool_experts": int(len(pool)),
        "sequences": S, "kmeans_score": score, "atlas_order_check": "per-domain counts equal the atlas meta",
        "seq_doc_ambiguous": ambiguous if meta.get("seq_docs") else None,
        "clusters": clusters, "labels_layer_expert": [[int(l), int(e), int(c)] for (l, e), c in zip(pool, lab)]}, indent=1))
    (out / "boxes.json").write_text(json.dumps(boxes, indent=1))
    if a.tagging_out:     # amoeba_topics_label.py inputs: pool [P, 2] (layer, unit) + labels [P]; one item per sequence
        np.savez(out / "pool_clusters.npz", pool=pool.astype(np.int64), labels=lab.astype(np.int64))
        items = [{"seq": t, "probe_id": rows[i].get("probe_id"), "topics": rows[i].get("topics"),
                  "split": rows[i].get(a.fit_field) if a.fit_field else "fit", "domain": dn}
                 for t, (i, dn) in enumerate(order)]
        (out / "items.json").write_text(json.dumps(items))
        (out / "TAGGING.md").write_text(
            "Tagging inputs (scripts/amoeba_topics_label.py): responses = the atlas peak_L*.npy rows (per-sequence "
            "peak |unit - mean|, one 128-token window per probe) in items.json order (first len(items) rows); "
            "pool_clusters.npz = pool (layer, expert) + labels; clusters were formed on fit sequences only, "
            "held sequences are for confirmation.\n")
    print(json.dumps({"candidates": sum(b["kind"] == "cluster" for b in boxes),
                      "support": sum(1 for c in clusters if str(c["verdict"]).startswith("support")),
                      "small": sum(1 for c in clusters if str(c["verdict"]).startswith("<"))}))


if __name__ == "__main__":
    main()
