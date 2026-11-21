#!/usr/bin/env python3
"""Wikipedia category-membership crawl for the 2026-11-18 category corpora (encyclopedic register).

The TF-IDF centroid used for subject-boxes-99a59a894 failed the blind judge for hum.history, life.sports,
life.travel and soc.education. Here a node's articles are the members of its Wikipedia categories (breadth-first,
depth <= --depth, at most --per-node titles). Membership only nominates; the term log-odds filter
(amoeba_category_mine.py) decides. Article text comes from the local enwiki 20231101 extract, so only title lists
are fetched (MediaWiki API, one request at a time, maxlag=5).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOTS = {
    "hum.history": ["History by period", "Historiography", "History by country"],
    "hum.history.ancient": ["Ancient history", "Ancient Rome", "Ancient Greece", "Ancient Egypt"],
    "hum.history.medieval": ["Middle Ages", "Medieval history of Europe", "Crusades"],
    "hum.history.modern": ["Modern history", "History of the British Empire", "Cold War", "Industrial Revolution"],
    "hum.history.military": ["Military history", "Battles", "Wars"],
    "hum.religion": ["Theology", "Religious texts", "Christian theology", "Islamic theology", "Buddhist philosophy"],
    "hum.film": ["Film genres", "Film theory", "Filmmaking", "Cinematography", "Film criticism"],
    "life.sports": ["Sports rules and regulations", "Sports terminology", "Association football terminology",
                    "Baseball terminology", "Basketball terminology", "Tennis terminology", "Cricket terminology",
                    "Sports techniques", "Olympic sports"],
    "life.travel": ["Tourism", "Types of tourism", "Tourist attractions", "Travel", "Hospitality industry"],
    "life.personal_finance": ["Personal finance", "Retirement plans", "Mortgage", "Credit", "Taxation and investment"],
    "life.home_diy": ["Home improvement", "Woodworking", "Plumbing", "Gardening", "Building materials"],
    "soc.economics": ["Economic theories", "Macroeconomics", "Microeconomics", "Economics terminology"],
    "soc.economics.finance": ["Finance", "Financial markets", "Derivatives (finance)", "Corporate finance"],
    "soc.economics.accounting": ["Accounting", "Accounting terminology", "Auditing"],
    "soc.business": ["Management", "Business terms", "Strategic management", "Business models", "Organizational behavior"],
    "soc.business.marketing": ["Marketing", "Advertising", "Brand management", "Market research"],
    "soc.education": ["Education theory", "Pedagogy", "Educational psychology", "Curricula", "Educational assessment and evaluation"],
    "soc.anthropology": ["Anthropology", "Cultural anthropology", "Ethnography", "Kinship and descent", "Archaeological theory"],
    "soc.communication": ["Communication studies", "Journalism", "Mass media", "Rhetoric", "Public relations"],
    "eng.civil": ["Civil engineering", "Structural engineering", "Geotechnical engineering", "Bridges by structural type", "Transportation engineering"],
    "eng.mechanical": ["Mechanical engineering", "Machines", "Mechanisms (engineering)", "Engine technology", "Heat transfer"],
    "eng.chemical": ["Chemical engineering", "Unit operations", "Chemical process engineering", "Separation processes"],
    "eng.industrial": ["Industrial engineering", "Manufacturing", "Operations research", "Quality management", "Lean manufacturing"],
    "eng.biomedical": ["Biomedical engineering", "Prosthetics", "Biomaterials", "Medical imaging"],
    "eng.electrical": ["Electrical engineering", "Electronic circuits", "Electric power", "Electrical components"],
    "eng.signal_processing": ["Signal processing", "Digital signal processing", "Filter theory", "Telecommunication theory"],
    "sci.physics": ["Concepts in physics", "Physical quantities", "Physics theorems"],
    "sci.physics.classical_mechanics": ["Classical mechanics"], "sci.physics.electromagnetism": ["Electromagnetism"],
    "sci.physics.thermodynamics": ["Thermodynamics", "Statistical mechanics"], "sci.physics.quantum": ["Quantum mechanics"],
    "sci.physics.relativity": ["Theory of relativity"], "sci.physics.particle": ["Particle physics"],
    "sci.physics.nuclear": ["Nuclear physics"], "sci.physics.condensed_matter": ["Condensed matter physics"],
    "sci.physics.optics": ["Optics"], "sci.physics.fluids": ["Fluid dynamics"], "sci.physics.atomic": ["Atomic physics"],
    "sci.physics.plasma": ["Plasma physics"], "sci.physics.high_energy_theory": ["Quantum field theory", "String theory"],
    "sci.physics.nonlinear": ["Nonlinear systems", "Chaos theory"], "sci.physics.units": ["Units of measurement"],
    "math": ["Mathematical concepts", "Mathematical theorems"],
    "math.number_theory": ["Number theory"], "math.algebra": ["Abstract algebra"],
    "math.algebra.linear_algebra": ["Linear algebra"], "math.algebra.group_theory": ["Group theory"],
    "math.algebra.ring_theory": ["Ring theory", "Commutative algebra"], "math.algebra.category_theory": ["Category theory"],
    "math.algebra.representation_theory": ["Representation theory"], "math.geometry": ["Geometry"],
    "math.geometry.algebraic_geometry": ["Algebraic geometry"], "math.geometry.differential_geometry": ["Differential geometry"],
    "math.geometry.euclidean": ["Euclidean geometry"], "math.topology": ["Topology"],
    "math.topology.algebraic_topology": ["Algebraic topology"], "math.analysis": ["Mathematical analysis"],
    "math.analysis.real_complex": ["Real analysis", "Complex analysis"], "math.analysis.functional_analysis": ["Functional analysis"],
    "math.analysis.pde": ["Partial differential equations"], "math.analysis.ode": ["Ordinary differential equations"],
    "math.analysis.calculus": ["Calculus"], "math.analysis.harmonic": ["Harmonic analysis", "Fourier analysis"],
    "math.combinatorics": ["Combinatorics", "Graph theory"], "math.logic": ["Mathematical logic", "Set theory"],
    "math.probability": ["Probability theory", "Stochastic processes"], "math.statistics": ["Statistical theory", "Statistical methods"],
    "math.numerical_analysis": ["Numerical analysis"], "math.optimization": ["Mathematical optimization"],
    "math.dynamical_systems": ["Dynamical systems", "Ergodic theory"], "math.mathematical_physics": ["Mathematical physics"],
    "math.theoretical_cs": ["Theoretical computer science", "Computability theory", "Formal languages"],
}
API = "https://en.wikipedia.org/w/api.php"
PACE = 1.5
UA = "amoeba-corpus-research/1.0 (private research crawler; titles only)"


def members(cat, cont=None):
    q = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmlimit": "500",
         "cmtype": "page|subcat", "format": "json", "maxlag": "5"}
    if cont:
        q["cmcontinue"] = cont
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": UA})
    for t in range(8):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            if "error" in r:                       # maxlag etc.
                raise RuntimeError(r["error"].get("code"))
            return r
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After") or 0) or 60 * (t + 1)
            print(f"  HTTP {e.code} on {cat}; sleeping {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  {type(e).__name__} {e} on {cat}; retry", flush=True)
            time.sleep(10 * (t + 1))
    raise SystemExit(f"giving up on Category:{cat} (rate limit); rerun later, the crawl resumes per node")


def crawl(roots, depth, cap):
    # titles: insertion-ordered and distinct, so the cap counts distinct articles (one that sits in several
    # crawled categories no longer ends the crawl early)
    seen, titles, frontier = set(), {}, [(r, 0) for r in roots]
    while frontier and len(titles) < cap:
        cat, d = frontier.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        cont = None
        while True:
            r = members(cat, cont)
            for m in r.get("query", {}).get("categorymembers", []):
                if m["ns"] == 14 and d < depth:
                    frontier.append((m["title"].split(":", 1)[1], d + 1))
                elif m["ns"] == 0:
                    titles[m["title"]] = None
            cont = r.get("continue", {}).get("cmcontinue")
            time.sleep(PACE)
            if not cont or len(titles) >= cap:
                break
    return list(titles)[:cap]


def node_settings(node, depth, per_node):
    return {"depth": depth, "per_node": per_node, "roots": list(ROOTS[node])}


def resume_conflicts(res, nodes_meta, depth, per_node):
    """Finished nodes whose recorded crawl settings differ from this invocation's: {node: {field: (recorded,
    current)}}. A finished node without a record (crawled before per-node records existed) is not a conflict;
    main() keeps it with a warning."""
    bad = {}
    for node in res:
        rec = nodes_meta.get(node)
        if node not in ROOTS or not rec or rec.get("unrecorded"):
            continue
        cur = node_settings(node, depth, per_node)
        diff = {k: (rec.get(k), v) for k, v in cur.items() if rec.get(k) != v}
        if diff:
            bad[node] = diff
    return bad


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--per-node", type=int, default=12000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    mpath = out.with_suffix(".meta.json")
    res = json.loads(out.read_text()) if out.exists() else {}
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    nodes = dict(old.get("nodes") or {})
    # A finished node is kept only under the settings it was crawled with: the meta file is the provenance of
    # every title list, so a rerun with another depth / cap / root list must not relabel old lists.
    if bad := resume_conflicts(res, nodes, a.depth, a.per_node):
        raise SystemExit("refusing to resume: finished nodes were crawled with other settings (recorded, current): "
                         + "; ".join(f"{n}: " + ", ".join(f"{k} {v[0]!r} != {v[1]!r}" for k, v in d.items())
                                     for n, d in bad.items())
                         + ". Rerun with the recorded settings or use another --out.")
    for node in res:
        if node in ROOTS and node not in nodes:
            print(f"WARNING {node}: finished before per-node crawl records existed; its depth / per-node cap / "
                  "roots are unverified", flush=True)
            nodes[node] = {"unrecorded": True, "note": "crawled before per-node records; settings unverified",
                           "legacy_meta": {k: old.get(k) for k in ("depth", "per_node", "fetched") if k in old}}

    def meta():
        return {"api": API, "fetched": time.strftime("%Y-%m-%d"), "depth": a.depth, "per_node": a.per_node,
                "roots": ROOTS, "nodes": nodes,
                "note": "top-level depth / per_node / roots / fetched describe the last invocation; nodes.<node> "
                        "records what each title list was crawled with"}

    for node, roots in ROOTS.items():
        if node in res:
            continue
        titles = crawl(roots, a.depth, a.per_node)
        res[node] = titles
        nodes[node] = node_settings(node, a.depth, a.per_node) | {"fetched": time.strftime("%Y-%m-%d"),
                                                                 "titles": len(titles)}
        write_atomic(mpath, json.dumps(meta(), indent=1))  # record first: a list never lacks its settings
        write_atomic(out, json.dumps(res))
        print(f"{node}: {len(res[node])} titles", flush=True)
    write_atomic(mpath, json.dumps(meta(), indent=1))


if __name__ == "__main__":
    main()
