#!/usr/bin/env python3
"""Tier carving corpora from English core corpus v2 (graded cores plan, docs/research/amoeba-27b/GRADED_CORES_PLAN.md).

CPU only. Writes the jsonl form `research.amoeba.carve` reads ({"text", "domain", "source", "split"}):
- one level (T1 = L1 grade school, T2 = L2 high school), prose registers only (narrative, expository,
  dialogue, instructions, technical); the literary registers (poetry, song, drama, letters) are T5's;
- domain = register;
- held rows keep split "held" (the measurement half); fit rows drop the split, so the driver hashes them
  by source group into mean / fit / spare (it has no mean split otherwise).
"""
import argparse
import hashlib
import json
from pathlib import Path

PROSE = ("narrative", "expository", "dialogue", "instructions", "technical")

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--level", required=True, choices=["L1", "L2", "L3"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(x) for x in open(a.corpus)]
out, counts = [], {}
for r in rows:
    if r["level"] != a.level or r["register"] not in PROSE:
        continue
    o = {"text": r["text"], "domain": r["register"], "source": r.get("group") or r["source"]}
    if r["split"] == "held":
        o["split"] = "held"
    out.append(o)
    k = (r["register"], r["split"])
    counts[f"{k[0]}:{k[1]}"] = counts.get(f"{k[0]}:{k[1]}", 0) + 1
p = Path(a.out)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
manifest = {"schema": "mlx2.amoeba-tier-corpus.v1", "source_corpus": a.corpus,
            "source_sha256": hashlib.sha256(Path(a.corpus).read_bytes()).hexdigest(), "level": a.level,
            "registers": PROSE, "rows": len(out), "counts": counts,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
p.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(manifest))
