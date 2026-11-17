#!/usr/bin/env python3
"""Tag, sort and carve auto-detected clusters into logical buckets, as a partition (CPU, no model).

Standard strategy (Pierre, 2026-11-14): auto-detect clusters, then the corpus, then tag, carve and
tile. Input: an amoeba27b_atlas.py output dir (pool_clusters.npz/json, whose source profiles are over
the capture's cells) and the export core. Each cluster gets family scores (the mean relative
activation over the cells of each family) and goes to the family with the highest score, if that score
is at least --min-selectivity times the median over families. Otherwise it goes to "general". Every
pool neuron lands in exactly one bucket and the core keeps its own, so core + buckets = every neuron
once ([[carve-is-a-partition]]). Writes buckets.npz (one [L, N] mask per bucket), buckets.json (tags,
sizes, layer spans, member clusters) and a coverage check.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

import numpy as np

FAMILIES = {
    "code": ["code:*", "v1:repo_markdown"],
    # logic and argument, one family per taxonomy v1.3 node (topics session, 2026-11-15)
    "logic": ["logic:logic"],
    "debate": ["logic:debate"],
    "socratic": ["logic:socratic"],
    "deduction": ["logic:rlvr_worlds"],
    "entailment": ["task:logic_entail"],
    "epistemic": ["logic:epistemic"],
    "math": ["task:arithmetic", "task:multiplication"],
    "languages": ["task:french"],
    "geography_facts": ["task:capitals"],
    "recall": ["task:recall"],
    "dialogue": ["cosmopedia:*:dialogue"],
    "science": ["domain:science_abstract", "cosmopedia:researchers:scientific"],
    "legal": ["domain:legal_bill"],
    "textbook": ["cosmopedia:*:textbook"],
    "narrative": ["tinystories", "v1:tinystories_train", "cosmopedia:young_children:*", "cosmopedia:children:*",
                         "cosmopedia:*:story"],
    "howto": ["cosmopedia:*:wikihow"],
    "encyclopedic": ["v1:donor_generated"],
}

# Tile names use the amoeba topic taxonomy (schema mlx2.amoeba-topics.v1, branch claude/amoeba-topics-20261114):
# tile = the family's topic ID, subtile = the topic of the cluster's strongest cell inside its family, when that
# is a finer node. Full prefixed form (Pierre, 2026-11-15): <model>.amoeba.v1.tile.<topic id>. None = no node
# yet (taxonomy gap: the topics session owns new IDs); those tiles are named <model>.amoeba.v1.tile.unassigned.<family>.
FAMILY_TOPIC = {
    "code": "prog", "logic": "hum.philosophy.logic", "debate": "form.debate", "socratic": "form.dialogue.socratic",
    "deduction": "skill.deduction", "entailment": "skill.entailment", "epistemic": "hum.philosophy.epistemology", "math": "math.arithmetic", "languages": "lang.fra",
    "geography_facts": "soc.geography.world_capitals", "recall": "skill.in_context_recall", "dialogue": "form.dialogue",
    "science": "sci", "legal": "soc.law.us_legislation", "textbook": "form.textbook", "narrative": "form.narrative",
    "howto": "form.instructions", "encyclopedic": "form.encyclopedic", "general": None,
}
CELL_TOPIC = [
    ("code:cpp", "prog.cpp"), ("code:csharp", "prog.csharp"), ("code:cuda", "prog.gpu.cuda"), ("code:java", "prog.java"),
    ("code:php", "prog.php"), ("code:python", "prog.python"), ("code:rust", "prog.rust"), ("code:shell", "prog.shell"),
    ("code:swift", "prog.swift"), ("code:typescript", "prog.typescript"), ("v1:repo_markdown", "prog.markup.markdown"),
    ("task:arithmetic", "math.arithmetic.addition"), ("task:multiplication", "math.arithmetic.multiplication"),
    ("tinystories", "form.narrative.childrens_story"), ("v1:tinystories_train", "form.narrative.childrens_story"),
    ("cosmopedia:young_children:*", "form.narrative.childrens_story"), ("cosmopedia:children:*", "form.narrative.childrens_story"),
]


# audience is a separate axis from subject and form (taxonomy v1.3): tagged on every cluster whose strongest
# in-family cell is a Cosmopedia audience cell
AUDIENCE = {"young_children": "form.audience.young_children", "children": "form.audience.children",
            "middle_school_students": "form.audience.middle_school", "high_school_students": "form.audience.high_school",
            "college_students": "form.audience.college", "college_studnets": "form.audience.college",
            "researchers": "form.audience.researchers"}


def audience_of(cell):
    parts = (cell or "").split(":")
    return AUDIENCE.get(parts[1]) if len(parts) == 3 and parts[0] == "cosmopedia" else None


def model_slug(checkpoint):
    """'Qwen3.8-27B-BF16-q8-mlx2@2dd8d304' -> 'qwen3_8_27b_bf16_q8_mlx2' (one dotted-path segment; the
    fingerprint stays in its own field)."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", checkpoint.split("@")[0].lower()).strip("_")


def tile_ids(slug, family, cell):
    top = FAMILY_TOPIC.get(family)
    if top is None:
        t = f"{slug}.amoeba.v1.tile.unassigned.{family}"
        return None, None, t, t
    sub = next((tp for pat, tp in CELL_TOPIC if fnmatch.fnmatch(cell or "", pat)), None)
    if not (sub and sub.startswith(top + ".")):
        sub = None
    tile = f"{slug}.amoeba.v1.tile.{top}"
    return top, sub, tile, (f"{slug}.amoeba.v1.tile.{sub}" if sub else tile)


def canon(cell):
    """Atlas v1 cell names (task_capitals, tinystories_train, ...) in the enriched capture's spelling."""
    if cell.startswith("task_"):
        return "task:" + cell[5:]
    if ":" not in cell and cell not in ("tinystories",):
        return "v1:" + cell
    return cell


def family_of_cells(cells):
    out = {}
    for fam, pats in FAMILIES.items():
        out[fam] = [c for c in cells if any(fnmatch.fnmatch(c, p) for p in pats)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="amoeba27b_atlas.py output dir")
    ap.add_argument("--core-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-selectivity", type=float, default=1.25)
    ap.add_argument("--taxonomy", help="taxonomy.v1.json; when given, every topic ID used must exist in it")
    ap.add_argument("--family-score", choices=["max", "mean"], default="max",
                    help="max: a family scores its strongest cell (default since 2026-11-15). mean averaged over the "
                         "family's cells, which favoured one-cell families (legal) over many-cell ones (science, code)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cd = Path(a.clusters)
    blob = np.load(cd / "pool_clusters.npz")
    info = json.loads((cd / "pool_clusters.json").read_text())
    pool, labels = blob["pool"].astype(np.int64), blob["labels"].astype(np.int64)
    core = np.load(Path(a.core_run) / "export_core.npz")["core"] > 0
    L, N = core.shape
    checkpoint = info.get("checkpoint") or "unknown"
    slug = model_slug(checkpoint)
    raw_cells = sorted(info["clusters"][0]["source_rel"])
    name = {c: canon(c) for c in raw_cells}
    fams = {f: [r for r in raw_cells if name[r] in cs] for f, cs in family_of_cells([name[r] for r in raw_cells]).items()}
    missing = [f for f, cs in fams.items() if not cs]
    rows, assign = [], {}
    for c in info["clusters"]:
        j = int(c["id"].split("C")[-1])
        rel = c["source_rel"]
        agg = np.max if a.family_score == "max" else np.mean
        scores = {f: float(agg([rel[x] for x in cs])) for f, cs in fams.items() if cs}
        if not scores:
            assign[j] = "general"
            topic, subtopic, tile, subtile = tile_ids(slug, "general", None)
            rows.append({"cluster": c["id"], "size": c["size"], "bucket": "general", "topic_id": None, "subtopic_id": None,
                         "tile_id": tile, "subtile_id": subtile, "selectivity": None,
                         "family_scores": {}, "top_cells": [], "top_tokens": c["top_tokens"][:8], "layers": c["layers"]})
            continue
        med = float(np.median(list(scores.values())))
        best = max(scores, key=scores.get)
        bucket = best if scores[best] >= a.min_selectivity * med else "general"
        assign[j] = bucket
        bcell = name[max(fams[bucket], key=lambda x: rel[x])] if bucket != "general" else None
        topic, subtopic, tile, subtile = tile_ids(slug, bucket, bcell)
        rows.append({"cluster": c["id"], "size": c["size"], "bucket": bucket, "topic_id": topic, "subtopic_id": subtopic,
                     "tile_id": tile, "subtile_id": subtile, "strongest_family_cell": bcell,
                     "audience_id": audience_of(bcell),
                     "selectivity": round(scores[best] / max(med, 1e-9), 3), "family_scores": {k: round(v, 3) for k, v in scores.items()},
                     "top_cells": sorted(rel, key=rel.get, reverse=True)[:5], "top_tokens": c["top_tokens"][:8],
                     "layers": c["layers"]})
    buckets = sorted(set(assign.values()))
    masks = {b: np.zeros((L, N), np.uint8) for b in buckets}
    for j, b in assign.items():
        sel = pool[labels == j]
        masks[b][sel[:, 0], sel[:, 1]] = 1
    total = core.astype(np.int64) + sum(m.astype(np.int64) for m in masks.values())
    coverage = {"neurons": int(L * N), "covered_once": int((total == 1).sum()), "uncovered": int((total == 0).sum()),
                "covered_more_than_once": int((total > 1).sum())}
    coverage["partition"] = coverage["covered_once"] == coverage["neurons"]
    np.savez_compressed(out / "buckets.npz", **{f"{b}__mask": m for b, m in masks.items()})
    summary = {"schema": "mlx2.amoeba-buckets.v1", "clusters_dir": str(cd), "core_run": a.core_run,
               "rule": "partition: core + buckets = every neuron exactly once; carve core->box, never prune",
               "families": FAMILIES, "family_score": a.family_score, "families_without_cells": missing, "min_selectivity": a.min_selectivity,
               "coverage": coverage,
               "checkpoint": checkpoint, "model_slug": slug, "taxonomy": "mlx2.amoeba-topics.v1",
               "tile_name_form": "<model>.amoeba.v1.tile.<topic id>",
               "buckets": {b: {"tile_id": tile_ids(slug, b, None)[2], "topic_id": FAMILY_TOPIC.get(b),
                               "taxonomy_gap": FAMILY_TOPIC.get(b) is None,
                               "subtiles": {st: int(sum(r["size"] for r in rows if r["bucket"] == b and r["subtile_id"] == st))
                                            for st in sorted({r["subtile_id"] for r in rows if r["bucket"] == b})},
                               "audiences": {au: int(sum(r["size"] for r in rows if r["bucket"] == b and r.get("audience_id") == au))
                                             for au in sorted({r.get("audience_id") for r in rows if r["bucket"] == b and r.get("audience_id")})},
                               "neurons": int(masks[b].sum()), "clusters": [r["cluster"] for r in rows if r["bucket"] == b],
                               "layers_with_neurons": int((masks[b].sum(1) > 0).sum())} for b in buckets},
               "clusters": rows}
    if a.taxonomy:
        known = {n["id"] for n in json.loads(Path(a.taxonomy).read_text())["nodes"]}
        used = {v for r in rows for v in (r.get("topic_id"), r.get("subtopic_id"), r.get("audience_id")) if v}
        bad = sorted(used - known)
        if bad:
            raise SystemExit(f"topic IDs not in the taxonomy: {bad}")
        summary["taxonomy_checked"] = {"file": a.taxonomy, "ids_used": sorted(used)}
    (out / "buckets.json").write_text(json.dumps(summary, indent=1))
    # the buckets in pool_clusters format, so amoeba27b_boxes.py --phases clusters tests each bucket
    # (add to core / remove from full, against a random twin) unchanged
    bidx = {b: i for i, b in enumerate(buckets)}
    blabels = np.array([bidx[assign[int(j)]] for j in labels], dtype=np.int16)
    np.savez_compressed(out / "pool_clusters.npz", pool=pool.astype(np.int32), labels=blabels)
    bc = []
    for b in buckets:
        mem = pool[blabels == bidx[b]]
        rs = [r for r in rows if r["bucket"] == b]
        bc.append({"id": f"bucket:C{bidx[b]}", "name": tile_ids(slug, b, None)[2], "bucket": b, "size": int(len(mem)),
                   "layers": {"min": int(mem[:, 0].min()), "max": int(mem[:, 0].max()), "median": int(np.median(mem[:, 0])),
                              "hist": np.bincount(mem[:, 0], minlength=L).tolist()},
                   "most_selective_source": b, "selectivity": max([r["selectivity"] or 0 for r in rs] or [0]),
                   "top_tokens": [t for r in rs[:3] for t in r["top_tokens"][:3]], "source_rel": {}})
    (out / "pool_clusters.json").write_text(json.dumps({"k": len(buckets), "pool_neurons": int(len(pool)),
                                                        "clusters": bc, "note": "buckets as clusters"}, indent=1))
    print(json.dumps({"coverage": coverage, "buckets": {b: v["neurons"] for b, v in summary["buckets"].items()}}, indent=1))
    if not coverage["partition"]:
        raise SystemExit("not a partition: check the core/pool inputs")


if __name__ == "__main__":
    main()
