"""Advanced-physics knowledge for ENS tiles (Pierre, 2026-11-23: "just do advanced topics from what we have in the
corpus, not just fact retrieval but also reasoning over it").

Source: the mathphys26 harvest (2026 arXiv theory physics, OAI metadata: title, abstract, categories; Pierre
2026-11-18). Papers from 2026 postdate Qwen3's training, so their facts are knowledge the base model cannot have.
Only rows whose license is CC BY, CC BY-SA or CC0 are used (the arXiv non-exclusive default and the non-commercial
licences are excluded). The derived fact set is labelled private_corpus and is not published.

Per paper (each fact a single Qwen3 token with a leading space):
  field     the primary category as a word (quant-ph -> " quantum", gr-qc -> " gravity", ...)
  month     the posting month from the arXiv id (2603.xxxxx -> " March")
  concept   a physics term of the abstract not in the title (see the rule below; 5+ letters, in 2+ sampled
            abstracts, single token). "Which paper is about <concept>" is a lookup over two records, the lightest of
            the reasoning tasks; same-field and earlier compare two records

Tasks built on them (physics_bus.py / physics_tile.py):
  retrieval  "'<title>' centres on the concept of" -> concept; "... was posted in" -> month; "... is a paper in" -> field
  reasoning  two papers, each mentioned by title: same field? (" Yes"/" No"); which was posted first? (" first"/
             " second"; same-month pairs excluded); which one is about <concept>? (" first"/" second")

  python -m research.ens.physics_facts --n 2000 --out <facts.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

HARVEST = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/downloads/mathphys26-oai-20261118"
FIELDS = {"quant-ph": "quantum", "cond-mat": "condensed", "gr-qc": "gravity", "hep-th": "strings",
          "hep-ph": "particle", "hep-lat": "lattice", "math-ph": "mathematical", "nucl-th": "nuclear",
          "nlin": "nonlinear"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
OK_LICENSES = ("creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/",
               "creativecommons.org/publicdomain/zero/")
TITLE_TOKENS_MAX = 24
SPECIFICITY_MIN = 50
# generic research verbs/nouns that pass the specificity bar but name no physics concept, and the field words
# (a concept equal to a field would make "which paper is about X" answerable from the field)
CONCEPT_STOP = {"derive", "derived", "derives", "compute", "computed", "parameter", "parameters", "scalable",
                "generalized", "generalize", "formulate", "characterize", "encode", "estimator", "estimation",
                "algorithm", "trivial", "negligible", "constrain", "constrained"}
GENERAL = Path.home() / "Desktop/amoeba-corpus/builds/lang-prose-eng-5cc2e1584/corpus.jsonl"   # general English


def _primary(cats):
    for c in cats:
        base = c.split(".")[0]
        if base in FIELDS:
            return base
    return None


def build(tok, n: int, seed: int = 0) -> dict:
    eng = {w.strip() for w in open("/usr/share/dict/words") if w.strip().islower()}
    one = {}

    def single(w):
        if w not in one:
            one[w] = len(tok.encode(" " + w)) == 1
        return one[w]
    pool = []
    for f in sorted(HARVEST.glob("physics_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            lic = r.get("license") or ""
            p = _primary(r.get("categories", []))
            if p is None or p != f.stem.removeprefix("physics_") or not any(s in lic for s in OK_LICENSES):
                continue
            title = re.sub(r"\s+", " ", r["title"]).strip()
            if len(tok.encode(" '" + title + "'")) > TITLE_TOKENS_MAX:
                continue
            mm = int(r["id"][2:4])
            if not (r["id"].startswith("26") and 1 <= mm <= 12):
                continue
            pool.append({"id": r["id"], "title": title, "field": FIELDS[p], "month": MONTHS[mm - 1],
                         "abstract": r["abstract"], "license": lic})
    rng = random.Random(seed)
    rng.shuffle(pool)
    by_field = {}
    for r in pool:
        by_field.setdefault(r["field"], []).append(r)
    per = max(1, n // len(by_field))
    sample = []
    for fld in sorted(by_field):                      # stratified by field, as even as the pool allows
        sample += by_field[fld][:per]
    sample = sample[:n]
    # concept: a physics term of the abstract, not in the title. Specificity = how much more often the word occurs in
    # physics abstracts (all harvested rows) than in general English prose; among words at least 20x more specific
    # to physics, take the one with the highest IDF across the sample (the most distinctive for this paper).
    gen = Counter()
    for line in GENERAL.open():
        gen.update(re.findall(r"[a-z]{5,}", json.loads(line)["text"].lower()))
    gen_total = sum(gen.values())
    phys = Counter()
    for r in pool:
        phys.update(re.findall(r"[a-z]{5,}", r["abstract"].lower()))
    phys_total = sum(phys.values())

    def specificity(w):
        return (phys[w] / phys_total) / ((gen[w] + 1) / gen_total)
    words = [set(re.findall(r"[a-z]{5,}", r["abstract"].lower())) for r in sample]
    df = Counter(w for ws in words for w in ws)
    papers = []
    for r, ws in zip(sample, words):
        title_words = set(re.findall(r"[a-z]+", r["title"].lower()))
        cands = [w for w in ws if w in eng and not w.endswith("ly") and w not in title_words
                 and w not in CONCEPT_STOP and w not in FIELDS.values() and df[w] >= 2 and single(w)
                 and specificity(w) >= SPECIFICITY_MIN]
        if not cands:
            continue
        concept = max(cands, key=lambda w: (math.log(len(sample) / df[w]), w))
        papers.append({k: r[k] for k in ("id", "title", "field", "month", "license")} | {"concept": concept})
    for p in papers:
        for k in ("field", "month", "concept"):
            assert single(p[k]), (p["id"], k, p[k])
    blob = json.dumps(papers, sort_keys=True)
    return {"schema": "ens.physics_facts.v1", "source": str(HARVEST), "n": len(papers), "seed": seed,
            "label": "private_corpus: derived from arXiv metadata; never publish", "licenses_allowed": OK_LICENSES,
            "fields": FIELDS, "concept_rule": f"physics/general-English specificity >= {SPECIFICITY_MIN}, not -ly, not a field word or "
                            "in CONCEPT_STOP, then max IDF in the sample", "concept_stop": sorted(CONCEPT_STOP),
            "general_corpus": str(GENERAL), "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16], "papers": papers,
            "field_counts": dict(Counter(p["field"] for p in papers)),
            "month_counts": dict(Counter(p["month"] for p in papers))}


def main(argv=None):
    from research.ens import lm_bus as LB, qwen3 as Q
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    facts = build(LB.Tok(Q.snapshot_dir()), a.n, a.seed)
    Path(a.out).write_text(json.dumps(facts, indent=1))
    print(json.dumps({k: facts[k] for k in ("n", "sha256", "field_counts", "month_counts")}))


if __name__ == "__main__":
    main()
