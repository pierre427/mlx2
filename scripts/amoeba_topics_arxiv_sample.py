#!/usr/bin/env python3
"""Stratified sample of arXiv abstracts for the topic probe corpus (Pierre approved the download
2026-11-14). Source: huggingface gfissore/arxiv-abstracts-2021 @ e4c5fbd4dec8e46a5dc869216fe1c94cc585757a
(metadata CC0-1.0). Keeps id, primary category, title and abstract only; author and submitter names are
dropped. Primary category -> taxonomy node through the nodes' refs.arxiv codes plus the table below;
seeded reservoir of --per-node records per node. Writes OUT/arxiv_sample.jsonl and OUT/provenance.json.
Run with python -I (the input is downloaded data)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAXO = ROOT / "docs/research/amoeba-topics/taxonomy.v1.json"
EXTRA = {   # arXiv codes without an exact refs.arxiv match (old archives included)
    "hep-ph": "sci.physics.particle", "hep-ex": "sci.physics.particle", "hep-lat": "sci.physics.particle",
    "nucl-ex": "sci.physics.nuclear", "cond-mat.mes-hall": "sci.physics.condensed_matter",
    "cond-mat.mtrl-sci": "sci.materials", "cond-mat.str-el": "sci.physics.condensed_matter",
    "cond-mat.supr-con": "sci.physics.condensed_matter", "cond-mat.soft": "sci.physics.condensed_matter",
    "cond-mat.quant-gas": "sci.physics.condensed_matter", "cond-mat.dis-nn": "sci.physics.condensed_matter",
    "cond-mat.other": "sci.physics.condensed_matter", "supr-con": "sci.physics.condensed_matter",
    "physics.chem-ph": "sci.chemistry.physical", "chem-ph": "sci.chemistry.physical",
    "physics.ao-ph": "sci.earth.meteorology", "physics.geo-ph": "sci.earth.geology",
    "physics.bio-ph": "sci.biology.molecular", "physics.med-ph": "sci.medicine.clinical",
    "physics.atm-clus": "sci.physics.atomic", "atom-ph": "sci.physics.atomic", "plasm-ph": "sci.physics.plasma",
    "nlin.CD": "sci.physics.nonlinear", "nlin.SI": "sci.physics.nonlinear", "nlin.PS": "sci.physics.nonlinear",
    "nlin.AO": "sci.physics.nonlinear", "chao-dyn": "sci.physics.nonlinear", "solv-int": "sci.physics.nonlinear",
    "patt-sol": "sci.physics.nonlinear",
    "math.CV": "math.analysis.real_complex", "math.SP": "math.analysis.functional_analysis",
    "funct-an": "math.analysis.functional_analysis", "math.KT": "math.topology.algebraic_topology",
    "math.GN": "math.topology", "alg-geom": "math.geometry.algebraic_geometry", "dg-ga": "math.geometry.differential_geometry",
    "q-alg": "math.algebra.quantum_algebra",
    "cs.LG": "ml.foundations", "stat.ML": "ml.foundations", "cs.CV": "ml.deep_learning.vision", "eess.IV": "ml.deep_learning.vision",
    "cs.IR": "ml.deep_learning.embeddings", "cs.SD": "ml.deep_learning.speech", "cs.NE": "ml.foundations",
    "cmp-lg": "ml.nlp", "cs.AR": "cs.systems", "cs.OS": "cs.systems", "cs.SE": "swe", "cs.PF": "swe.performance",
    "cs.SY": "eng.electrical", "eess.SY": "eng.electrical", "stat.ME": "math.statistics", "stat.AP": "math.statistics",
    "stat.CO": "math.statistics", "q-fin.ST": "soc.economics.finance", "q-fin.GN": "soc.economics.finance",
    "q-fin.MF": "soc.economics.finance", "q-fin.PR": "soc.economics.finance", "q-fin.RM": "soc.economics.finance",
    "q-fin.CP": "soc.economics.finance", "q-fin.PM": "soc.economics.finance", "q-fin.TR": "soc.economics.finance",
    "q-fin.EC": "soc.economics", "econ.GN": "soc.economics", "econ.EM": "soc.economics", "econ.TH": "soc.economics",
    "q-bio.GN": "sci.biology.genetics", "q-bio.CB": "sci.biology.cell", "q-bio.MN": "sci.biology.molecular",
    "physics.soc-ph": "soc.sociology", "physics.ed-ph": "soc.education", "physics.hist-ph": "hum.history",
    "math.HO": "hum.history",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-node", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taxo = json.loads(TAXO.read_text())
    valid = {n["id"] for n in taxo["nodes"]}
    amap = {}
    for n in taxo["nodes"]:
        a = (n.get("refs") or {}).get("arxiv")
        if a and n["id"].split(".")[0] in ("sci", "math", "cs", "ml", "eng", "soc", "hum") and a not in amap:
            amap[a] = n["id"]
    amap.pop("cs.CL", None)
    amap["cs.CL"] = "ml.nlp"
    amap.update(EXTRA)
    for a, t in amap.items():
        assert t in valid, (a, t)
    rng = random.Random(args.seed)
    res, seen = defaultdict(list), defaultdict(int)
    unmapped = defaultdict(int)
    t0 = time.time()
    with gzip.open(args.src, "rt") as f:
        for line in f:
            r = json.loads(line)
            cats = r.get("categories")
            cats = cats[0] if isinstance(cats, list) else cats
            prim = str(cats).split()[0]
            node = amap.get(prim) or amap.get(prim.split(".")[0])
            if not node:
                unmapped[prim] += 1
                continue
            text = (r.get("title") or "").strip().replace("\n ", " ") + "\n" + " ".join((r.get("abstract") or "").split())
            rec = {"arxiv_id": r["id"], "primary": prim, "topic": node, "text": text}
            seen[node] += 1
            if len(res[node]) < args.per_node:
                res[node].append(rec)
            else:
                j = rng.randrange(seen[node])
                if j < args.per_node:
                    res[node][j] = rec
    n = 0
    with open(out / "arxiv_sample.jsonl", "w") as f:
        for node in sorted(res):
            for rec in sorted(res[node], key=lambda r: r["arxiv_id"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    prov = {"source": "huggingface gfissore/arxiv-abstracts-2021", "revision": "e4c5fbd4dec8e46a5dc869216fe1c94cc585757a",
            "license": "CC0-1.0 (arXiv metadata)", "approved": "Pierre 2026-11-14 (via coordinator session)",
            "src_sha256": hashlib.sha256(Path(args.src).read_bytes()).hexdigest(), "records_kept": n,
            "per_node": args.per_node, "seed": args.seed, "fields_kept": ["arxiv_id", "primary", "title+abstract"],
            "fields_dropped": "authors, submitter, comments, doi, versions", "nodes": {k: len(v) for k, v in sorted(res.items())},
            "population": dict(sorted(seen.items())), "unmapped_primary": dict(sorted(unmapped.items())),
            "built_s": round(time.time() - t0, 1)}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"kept {n} records over {len(res)} nodes; unmapped {sum(unmapped.values())}")


if __name__ == "__main__":
    main()
