#!/usr/bin/env python3
"""Tag an FN probe-atlas cluster set with taxonomy topics in one command (CPU).

Input: the 27B session's atlas dir (atlas/peak_L*.npy, atlas/meta.json; clusters/items.json, pool_clusters.npz) and
the probe corpus it was captured on. Steps:
  1. capture adapter (ARTIFACTS/fn-tag-capture-<atlas>): items.json (probe id, topics, split), resp_L = the first
     len(items) atlas rows in items.json order, meta (checkpoint, atlas commit, corpus sha), core = complement of pool
  2. amoeba_topics_label.py: per-cluster labels against 32 same-layer-count random twins, label-shuffle z-test with
     Bonferroni, fit chooses / held confirms; subclusters; per-neuron (expert) topic d maps
  3. per language-majority cluster: language ranking (z of the mean cluster score per language, all items) and
     the families those languages form (descriptive; family names are not taxonomy nodes)
  4. OUT/README.md summary
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True, help="fn-probeatlas-<rev> dir")
    ap.add_argument("--corpus", required=True, help="probe corpus.jsonl the atlas was captured on")
    ap.add_argument("--checkpoint", default="Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP@25c2da42")
    ap.add_argument("--atlas-commit", required=True)
    a = ap.parse_args()
    D = Path(a.atlas)
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cap = ART / f"fn-tag-capture-{D.name.split('-')[-1]}-{rev}"
    out = ROOT / f"docs/research/amoeba-topics/labels-fn-{D.name.split('-')[-1]}-{rev}"
    arrays = ART / f"labels-fn-{D.name.split('-')[-1]}-{rev}"
    cap.mkdir(parents=True, exist_ok=False)
    it = json.loads((D / "clusters/items.json").read_text())
    meta = json.loads((D / "atlas/meta.json").read_text())
    assert all(i["seq"] == k for k, i in enumerate(it)), "items not in atlas row order"
    S = len(it)
    (cap / "items.json").write_text(json.dumps([{"id": i["probe_id"], "topics": i["topics"], "split": i["split"],
                                                 "seq": i["seq"]} for i in it]))
    for l in range(meta["layers"]):
        x = np.load(D / f"atlas/peak_L{l:02d}.npy", mmap_mode="r")
        np.save(cap / f"resp_L{l:02d}.npy", np.asarray(x[:S]))
    z = np.load(D / "clusters/pool_clusters.npz")
    core = np.ones((meta["layers"], meta["neurons"]), np.int8)
    core[z["pool"][:, 0], z["pool"][:, 1]] = 0
    np.savez_compressed(cap / "core_complement_of_pool.npz", core=core)
    import hashlib
    corpus_sha = hashlib.sha256(Path(a.corpus).read_bytes()).hexdigest()
    (cap / "meta.json").write_text(json.dumps({
        "checkpoint": a.checkpoint, "layers": meta["layers"], "neurons": meta["neurons"], "sequences": S,
        "unit": "routed expert (layer, expert)", "response": "per-sequence peak |unit - mean| (27B session probeatlas)",
        "source_atlas": str(D), "mlx2_source_revision": a.atlas_commit[:9], "atlas_mlx2_commit": a.atlas_commit,
        "corpus": a.corpus, "corpus_sha256": corpus_sha, "core": "complement of the clusters' pool"}, indent=1))
    print(f"adapter: {S} sequences, {int(core.sum())} core units", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_label.py"), "--capture", str(cap),
                        "--corpus", a.corpus, "--out", str(out), "--arrays", str(arrays),
                        "--clusters", str(D / "clusters/pool_clusters.npz"), "--core", str(cap / "core_complement_of_pool.npz")],
                       cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}/scripts"})
    if r.returncode:
        raise SystemExit(f"labeller failed rc {r.returncode}")
    lab = json.loads((out / "labels.json").read_text())
    tax = {n["id"]: n["name"] for n in json.loads((ROOT / "docs/research/amoeba-topics/taxonomy.v1.json").read_text())["nodes"]}
    G = np.load(arrays / "cluster_group_scores.npy")
    ng = G.shape[1] // len(lab["clusters"])
    lang = np.array([next((t for t in i["topics"] if t.startswith("lang.")), "") for i in it])
    codes = [c for c, n in collections.Counter(lang).items() if c and n >= 20]
    desc = {}
    for c in lab["clusters"]:
        langs = [l["topic"] for l in c["labels"] if l["topic"].startswith("lang")]
        if not langs:
            continue
        j = int(c["id"].split("C")[1])
        s = G[:, j * ng]
        ml = {k: float(s[lang == k].mean()) for k in codes}
        mu, sd = np.mean(list(ml.values())), np.std(list(ml.values()))
        top = sorted(ml.items(), key=lambda kv: -kv[1])[:6]
        desc[c["id"]] = {"labels": langs, "experts": c["neurons"],
                         "top_languages": [{"id": k, "name": tax.get(k, k), "z": round((v - mu) / sd, 2)} for k, v in top]}
    (out / "lang_clusters_descriptive.json").write_text(json.dumps(
        {"note": "descriptive ranking of languages per language cluster (z of the mean cluster score over all items); "
                 "the test result is the label list (leaf labels pass the same gates as every node)", "clusters": desc},
        indent=1, ensure_ascii=False) + "\n")
    roots = collections.Counter(l["topic"].split(".")[0] for c in lab["clusters"] for l in c["labels"])
    leaf_lang = sorted({l["topic"] for c in lab["clusters"] for l in c["labels"] if l["topic"].startswith("lang.")})
    lines = [f"# FN cluster tags: {D.name} (labeller at mlx2 {rev})", "",
             f"- Checkpoint `{a.checkpoint}`; atlas mlx2 commit `{a.atlas_commit}`; corpus `{a.corpus}` (sha256 {corpus_sha[:16]}…).",
             f"- {lab['summary']}; labels by root: {dict(roots.most_common())}.",
             f"- Language leaves that passed the gates: {len(leaf_lang)} ({', '.join(leaf_lang[:40])}{' …' if len(leaf_lang) > 40 else ''}).",
             "- Per language cluster, the top languages (descriptive, z): see `lang_clusters_descriptive.json`.",
             "- Tags are statistical (fit-chosen, held-confirmed against random twins and a shuffled-label null); removal "
             "checks need a GPU window.", ""]
    for cid, d in sorted(desc.items(), key=lambda kv: int(kv[0].split("C")[1])):
        lines.append(f"- {cid} ({d['experts']}): {', '.join(d['labels'])} | " +
                     ", ".join(f"{x['name']} {x['z']}" for x in d["top_languages"][:5]))
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
