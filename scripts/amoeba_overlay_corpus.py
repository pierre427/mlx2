#!/usr/bin/env python3
"""Fit-side physics corpus for the donor-overlay capacity sweep (Pierre 2026-11-19 ~21:35: "using our entire corpus
of physics fine tuned into the host expert box overlay"). CPU only (tokenizer, no model).

Reads the fit rows of every box at or below --box-prefix in a category build (default the split-fixed
category-boxes-8ec12cb93, sha256(group) % 8: 0-4 fit), drops every row that could touch an evaluation text of the
bridge lab's physics run, cuts each kept document to its first --chunk-tokens host tokens and writes
OUT/chunks.jsonl and OUT/manifest.json (temp dir, then rename). Evaluation texts = the bridge probe corpus's
physics rows (topics at or below --topic) of the held split (the 200 spent dev texts and the 297 untouched
reserved final-test texts are both in it) and the run's 40 validation texts (named by the split manifest).

Disjointness, as the catboxes stage checks it (scripts/amoeba_held_purge.py / amoeba_heldfit_check.py): a fit row is
dropped when
  group     its group (routing suffix "#..." stripped) is the group of an evaluation text, or of any held-split
            probe row of any topic (whole held source groups stay out);
  text      its normalised text (whitespace collapsed, lower-cased) has the sha256 of an evaluation text's;
  span      it shares any --shingle-chars-long normalised character span with an evaluation text (probe texts are
            windows cut from documents, so an exact-text test alone would miss a containing document).
After the drop the three properties are re-checked on what is written; any hit exits 4 and writes nothing.
Exact duplicates (normalised text) inside the fit rows are kept once.

    amoeba_overlay_corpus.py --tokenizer ~/mlx-models/Studio-Operator-Qwen3.5-9B-4bit --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path.home() / "Desktop/amoeba-corpus/builds/category-boxes-8ec12cb93"
PROBES = Path.home() / ("Library/Application Support/mlx2/research-artifacts/amoeba/topic-probes-9a1f35475/"
                        "corpus.jsonl")
MANIFEST = ROOT / "docs/research/amoeba-bridge/physics-split-manifest-bridgeT2-8acf5a186.json"


def norm_text(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def norm_group(g):
    return str(g).split("#", 1)[0]


def under(t, prefix):
    return t == prefix or t.startswith(prefix + ".")


def shingles(t, k):
    """hashes of every k-character window of a normalised text (a text shorter than k: the whole text)"""
    if len(t) <= k:
        return {hash(t)}
    return {hash(t[i:i + k]) for i in range(len(t) - k + 1)}


def touches(t, shs, k):
    if len(t) <= k:
        return hash(t) in shs
    return any(hash(t[i:i + k]) in shs for i in range(len(t) - k + 1))


def eval_texts(probe_rows, man, topic):
    """(evaluation rows, groups held out): physics held rows + the manifest's validation ids; groups of every
    held-split probe row (any topic) plus the evaluation rows' groups"""
    val_ids = set((man.get("selection") or {}).get("val", []))
    ev = [r for r in probe_rows if (r["split"] == "held" and any(under(t, topic) for t in r["topics"]))
          or r["id"] in val_ids]
    groups = {norm_group(r["group"]) for r in probe_rows if r["split"] == "held"} | {norm_group(r["group"]) for r in ev}
    missing = val_ids - {r["id"] for r in ev}
    if missing:
        raise SystemExit(f"validation ids missing from the probe corpus: {sorted(missing)[:5]}")
    res = set((man.get("reserved_final_test") or {}).get("probe_ids", []))
    if not res <= {r["id"] for r in ev}:
        raise SystemExit("reserved final-test ids are not all physics held rows of the probe corpus")
    return ev, groups


def filter_rows(fit_rows, ev, groups, k):
    """kept rows (first of each normalised text) and drop counts by reason"""
    ev_sha = {hashlib.sha256(norm_text(r["text"]).encode()).hexdigest() for r in ev}
    shs = set()
    for r in ev:
        shs |= shingles(norm_text(r["text"]), k)
    kept, why, seen = [], Counter(), set()
    for r in fit_rows:
        t = norm_text(r["text"])
        h = hashlib.sha256(t.encode()).hexdigest()
        if norm_group(r["group"]) in groups:
            why["group"] += 1
        elif h in ev_sha:
            why["text"] += 1
        elif touches(t, shs, k):
            why["span"] += 1
        elif h in seen:
            why["duplicate"] += 1
        else:
            seen.add(h)
            kept.append(r)
    return kept, why, shs, ev_sha


def recheck(kept, groups, shs, ev_sha, k):
    bad = Counter()
    for r in kept:
        t = norm_text(r["text"])
        bad["group"] += norm_group(r["group"]) in groups
        bad["text"] += hashlib.sha256(t.encode()).hexdigest() in ev_sha
        bad["span"] += touches(t, shs, k)
    return {k_: int(v) for k_, v in bad.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=str(BUILD))
    ap.add_argument("--box-prefix", default="sci.physics")
    ap.add_argument("--probes", default=str(PROBES))
    ap.add_argument("--split-manifest", default=str(MANIFEST))
    ap.add_argument("--topic", default="sci.physics")
    ap.add_argument("--tokenizer", required=True, help="host model directory (its tokenizer)")
    ap.add_argument("--chunk-tokens", type=int, default=128)
    ap.add_argument("--shingle-chars", type=int, default=64)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = every kept document")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    if out.exists():
        sys.exit(f"refusing: {out} exists")
    t0 = time.time()
    build = Path(a.build).expanduser()
    bman = json.loads((build / "manifest.json").read_text())
    corpus = build / "corpus.jsonl.gz"
    probes = Path(a.probes).expanduser()
    man = json.loads(Path(a.split_manifest).read_text())
    probe_rows = [json.loads(x) for x in open(probes) if x.strip()]
    ev, groups = eval_texts(probe_rows, man, a.topic)
    fit, boxes = [], Counter()
    with gzip.open(corpus, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("split") == "fit" and under(str(r.get("box")), a.box_prefix):
                fit.append({k: r.get(k) for k in ("id", "box", "register", "group", "source", "license", "text")})
                boxes[r["box"]] += 1
    kept, why, shs, ev_sha = filter_rows(fit, ev, groups, a.shingle_chars)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path(a.tokenizer).expanduser()), local_files_only=True)
    chunks, short = [], 0
    for r in kept:
        ids = tok.encode(r["text"][:8 * a.chunk_tokens + 2000], add_special_tokens=False)
        if len(ids) < a.chunk_tokens + 8:  # the cut text's last tokens can split differently: keep a margin
            short += 1
            continue
        chunks.append({"id": r["id"], "box": r["box"], "register": r["register"], "group": r["group"],
                       "license": r["license"], "ids": ids[:a.chunk_tokens]})
        if a.max_docs and len(chunks) >= a.max_docs:
            break
    cids = {c["id"] for c in chunks}
    bad = recheck([r for r in kept if r["id"] in cids], groups, shs, ev_sha, a.shingle_chars)
    if any(bad.values()):
        print(f"disjointness re-check failed: {bad}", file=sys.stderr)
        return 4
    tmp = Path(str(out) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with open(tmp / "chunks.jsonl", "w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    rev = os.popen(f"git -C {ROOT} rev-parse --short=9 HEAD").read().strip()
    m = {"schema": "mlx2.amoeba-overlay-corpus.v1", "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "builder": "scripts/amoeba_overlay_corpus.py", "mlx2_source_revision": rev,
         "build": str(build), "build_corpus_sha256": bman.get("corpus_sha256"),
         "build_mlx2_revision": bman.get("mlx2_source_revision"), "box_prefix": a.box_prefix, "split": "fit only",
         "probes": str(probes), "probes_sha256": hashlib.sha256(probes.read_bytes()).hexdigest(),
         "split_manifest": str(a.split_manifest),
         "split_manifest_sha256": hashlib.sha256(Path(a.split_manifest).read_bytes()).hexdigest(),
         "tokenizer": str(Path(a.tokenizer).expanduser().resolve()), "chunk_tokens": a.chunk_tokens,
         "tokenizer_sha256": (hashlib.sha256((Path(a.tokenizer).expanduser() / "tokenizer.json").read_bytes())
                              .hexdigest() if (Path(a.tokenizer).expanduser() / "tokenizer.json").exists() else None),
         "rule": ("fit rows of boxes at or below box_prefix; dropped: group (suffix stripped) of any held-split probe "
                  "row or evaluation text, normalised-text sha256 of an evaluation text, any shared "
                  f"{a.shingle_chars}-character normalised span with an evaluation text, exact duplicates; each kept "
                  f"document's first {a.chunk_tokens} host tokens (documents shorter dropped)"),
         "evaluation_texts": {"physics_held_rows": sum(r["split"] == "held" for r in ev), "val_rows":
                              sum(r["split"] != "held" for r in ev), "held_groups_any_topic": len(groups)},
         "fit_rows_by_box": dict(boxes), "fit_rows": len(fit), "dropped": dict(why), "short": short,
         "chunks": len(chunks), "tokens": len(chunks) * a.chunk_tokens,
         "registers": dict(Counter(c["register"] for c in chunks)),
         "licenses": dict(Counter(str(c["license"]) for c in chunks).most_common(12)),
         "recheck": bad, "wall_s": round(time.time() - t0, 1)}
    m["chunks_sha256"] = hashlib.sha256((tmp / "chunks.jsonl").read_bytes()).hexdigest()
    (tmp / "manifest.json").write_text(json.dumps(m, indent=1))
    tmp.rename(out)
    print(json.dumps({k: m[k] for k in ("fit_rows", "dropped", "short", "chunks", "tokens", "recheck", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
