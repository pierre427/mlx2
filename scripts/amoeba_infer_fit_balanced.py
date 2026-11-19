#!/usr/bin/env python3
"""Source-balanced fit build for the FN inference boxes (ml.inference / ml.llm / ml.deep_learning). CPU only.

Why (2026-11-18): the FN firing capture takes the first --max-docs rows of a seeded shuffle, so the infer-fit-2e4acb375
fit rows (about 80% code and PR text) gave arxiv, trace, tuning and wiki only 4-18 documents per half. With so few
documents those sources cannot reach significance, and amoeba_fn_partition.py --require-every-source then removed
nearly all of the ml boxes (ml.inference 57,565 box units -> 713). Not a shortage of text: the probe corpus has 90 to
1,100 rows of at least 192 tokens per small source and box.

What it writes: per box a JSONL with
  fit rows: up to --per-source documents for each source (the `source` field before ':'), each at least --min-tokens
            long. A row that is long enough is a document on its own; shorter rows are joined with the next short rows
            of the same source ("\\n\\n") until the window is long enough; the window lists its rows in `parts`. A capture with --max-docs >= the build size
            therefore uses every document and every source equally.
  held rows: the held rows of --base-fit for the box, unchanged, so the box battery's held set does not move.
Rows come from --probes (split fit only), excluding every group (source repo or paper) that has a held row in
--base-fit or a non-fit row in --probes, so fit and held share no repo. A row is used for a box only if its topics hit that box and no other of
the three (the boxes stay disjoint, as in the base build).

Usage: amoeba_infer_fit_balanced.py --probes CORPUS.jsonl --base-fit BUILD_DIR --out BUILD_DIR
       --tokenizer tokenizer.json [--per-source 160] [--min-tokens 224] [--seed 0]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BOXES = ("ml.inference", "ml.llm", "ml.deep_learning")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def counter(path):
    if path == "words":                                   # tests only
        return lambda s: len(s.split())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    return lambda s: len(tok.encode(s, add_special_tokens=False).ids)


def box_of(topics):
    hit = {b for b in BOXES for t in topics if t == b or t.startswith(b + ".")}
    return hit.pop() if len(hit) == 1 else None


def merge(parts, box, n):
    text = "\n\n".join(r["text"] for r in parts)
    red = {r.get("redistribute") for r in parts}
    first = parts[0]
    return {"id": f"bal-{box}-{n:05d}", "topics": sorted({t for r in parts for t in r["topics"]}), "split": "fit",
            "text": text, "origin": first["origin"] if len({r["origin"] for r in parts}) == 1 else "mixed",
            "source": first["source"], "source_revision": "; ".join(sorted({str(r.get("source_revision")) for r in parts})),
            "group": first["group"] if len({r["group"] for r in parts}) == 1 else "mixed", "license": "; ".join(sorted({str(r.get("license")) for r in parts})),
            "redistribute": "no" if "no" in red else ("unresolved" if red - {"yes"} else "yes"),
            "label_method": "; ".join(sorted({str(r.get("label_method")) for r in parts})),
            "parts": [r["id"] for r in parts], "sha256": hashlib.sha256(text.encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--base-fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the captured model ('words': whitespace, tests)")
    ap.add_argument("--per-source", type=int, default=160)
    ap.add_argument("--min-tokens", type=int, default=224, help="margin over the capture's --tokens 192")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing existing {out}")
    base = Path(a.base_fit)
    held, held_groups = {}, set()
    for b in BOXES:
        held[b] = [json.loads(x) for x in (base / f"{b}.jsonl").read_text().splitlines() if x.strip()]
        held[b] = [r for r in held[b] if r["split"] == "held"]
        held_groups |= {r["group"] for r in held[b]}
    ntok = counter(a.tokenizer)
    probes = [json.loads(x) for x in open(a.probes) if x.strip()]
    held_groups |= {r.get("group") for r in probes if r.get("split") != "fit"}
    cand = collections.defaultdict(list)
    for r in probes:
        if r.get("split") != "fit" or r.get("group") in held_groups:
            continue
        b = box_of(r.get("topics") or [])
        if b:
            cand[(b, r["source"].split(":", 1)[0])].append(r)
    rng = np.random.default_rng(a.seed)
    manifest = {"schema": "mlx2.amoeba-infer-fit-balanced.v1", "boxes": {}, "params": {
        "per_source": a.per_source, "min_tokens": a.min_tokens, "seed": a.seed,
        "tokenizer": a.tokenizer, "tokenizer_sha256_12": None if a.tokenizer == "words" else sha256_file(a.tokenizer)[:12]},
        "inputs": {"probes": a.probes, "probes_sha256": sha256_file(a.probes), "base_fit": str(base),
                   "base_fit_manifest_sha256": sha256_file(base / "manifest.json")},
        "held_groups_excluded": len(held_groups),
        "note": "fit: per-source balanced, short rows joined within a source; held: the base build's held rows unchanged"}
    here = Path(__file__).resolve().parent
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=here).stdout.strip()
    if dirty and a.tokenizer != "words":
        raise SystemExit("refusing: dirty mlx2 tree (the build records its revision)")
    out.mkdir(parents=True)
    try:
        manifest["mlx2_source_revision"] = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True,
                                                          text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except OSError:
        manifest["mlx2_source_revision"] = None
    for b in BOXES:
        rows, per = [], {}
        for src in sorted(s for (bb, s) in cand if bb == b):
            pool = cand[(b, src)]
            order = rng.permutation(len(pool))
            docs, buf, used, buf_n = [], [], 0, 0
            for j in order:
                if len(docs) >= a.per_source:
                    break
                r = pool[j]
                n = ntok(r["text"])
                used += 1
                if n >= a.min_tokens:                  # long enough alone, even while a short buffer is open
                    docs.append([r])
                    continue
                buf.append(r)
                buf_n += n
                if buf_n >= a.min_tokens:
                    docs.append(buf)
                    buf, buf_n = [], 0
            for parts in docs:
                rows.append(merge(parts, b, len(rows)))
            per[src] = {"docs": len(docs), "joined": sum(len(p) > 1 for p in docs), "rows_used": used,
                        "rows_available": len(pool), "short_rows_left_unjoined": len(buf)}
        with open(out / f"{b}.jsonl", "w") as f:
            for r in rows + held[b]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["boxes"][b] = {"fit_docs": len(rows), "held_rows": len(held[b]), "per_source": per,
                                "sha256": sha256_file(out / f"{b}.jsonl")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({b: {"fit": v["fit_docs"], "held": v["held_rows"],
                          "per_source": {s: x["docs"] for s, x in v["per_source"].items()}}
                      for b, v in manifest["boxes"].items()}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
