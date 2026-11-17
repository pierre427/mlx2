#!/usr/bin/env python3
"""Build a model's topic map (schema mlx2.amoeba-topic-map.v1) from a labelling run, its plan and the
functional check (CPU, no model). DESIGN: docs/research/amoeba-topics/DESIGN.md.

Writes docs/research/amoeba-topics/registry/<slug>.topics.json and <slug>.topics.npz (neuron maps:
key "<topic>|<unit>" -> int32 [n, 2] (layer, neuron)). One row per tested taxonomy node, plus a row
for every topic held by a registered box.

Row status:
  cluster        a cluster/subcluster labelled X whose removal costs X more than its twin and more than
                 it costs the off-topic control (functional gate below)
  box            a carved box tagged X (neurons chosen by mask carving; weights unchanged) (box registry / 27B boxes evidence); listed beside any cluster
  cluster_unverified  labelled units exist but none was functionally measured (budget-stopped run)
  subsumed       a descendant of X has a passing unit (reason subsumed_by:<descendant>)
  core_resident  X is selective in the model but no unit passes: its neural map is the core neurons
                 selective for X on fit (the plan's core map, chance-controlled), with reason_not_split
  absent         X was tested, no unit and no core selectivity above chance
Functional gate: remove_over_twin(X) (vs the twins' mean) >= --min-effect nats, the unit beats every twin,
and the effect is >= 2 x |remove_over_twin(control)|.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import amoeba_topics_lib as T

SCHEMA = "mlx2.amoeba-topic-map.v1"
REG = T.ROOT / "docs/research/amoeba-topics/registry"


def slug(ckpt: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", ckpt.replace("/", "--"))


def functional_pass(fr, topic, min_effect):
    if not fr:
        return None, "not measured"
    t = fr["topics"].get(topic)
    ctl = next((v for k, v in fr["topics"].items() if k.startswith("control(")), None)
    if not t or "remove_over_twin" not in t:
        return None, "not measured"
    if not ctl or "remove_over_twin" not in ctl or "remove_over_twin_max" not in t:
        return None, "not measured (control or twin set missing)"
    eff = t["remove_over_twin"]                     # unit minus the mean of its twins
    worst = t["remove_over_twin_max"]                # unit minus the strongest twin
    c = abs(ctl["remove_over_twin"])
    ok = eff >= min_effect and worst > 0 and eff >= 2 * c
    why = None if ok else (f"remove_over_twin {eff:.3f} < {min_effect}" if eff < min_effect else
                           f"a twin costs as much ({worst:.3f})" if worst <= 0 else f"not topic-specific (control {c:.3f})")
    return ok, why


def per_layer(pairs, d=None):
    out = {}
    for l in np.unique(pairs[:, 0]):
        m = pairs[:, 0] == l
        e = {"n": int(m.sum())}
        if d is not None:
            e["d_held_mean"] = round(float(np.mean(d[m])), 3)
        out[str(int(l))] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--function", default="", help="comma list of function.json dirs (absent: 'not measured')")
    ap.add_argument("--boxes", default="", help="json list of {topic, box_id, ref, neurons} for carved boxes")
    ap.add_argument("--out-dir", default=str(REG))
    ap.add_argument("--min-effect", type=float, default=0.02)
    ap.add_argument("--core-d", type=float, default=1.0)
    ap.add_argument("--size-floor", type=int, default=256)
    ap.add_argument("--english-overlap", type=float, default=0.8)
    ap.add_argument("--rollup-tol", type=float, default=0.01, help="nats a member may lose inside a roll-up")
    ap.add_argument("--core-corpus", default="", help="comma list of the corpus domains the core was carved on")
    ap.add_argument("--by-construction", default="prog,swe,ml,sec,cs",
                    help="branches whose core residency may follow from carving the core on repo-style text")
    args = ap.parse_args()
    lab = json.loads((Path(args.labels) / "labels.json").read_text())
    plan = json.loads((Path(args.plan) / "plan.json").read_text())
    U = np.load(Path(args.plan) / "units.npz")
    fn = {"results": [], "mlx2_source_revision": []}
    for fdir in filter(None, args.function.split(",")):           # several partial runs merge, later wins
        f = json.loads((Path(fdir) / "function.json").read_text())
        fn["results"] += f["results"]
        fn["mlx2_source_revision"].append(f.get("mlx2_source_revision"))
    fres = {r["unit"]: r for r in fn["results"]}
    core = np.load(plan["core"])["core"] > 0
    L, N = core.shape
    tax = T.Taxonomy()
    node_ids = json.loads((Path(args.arrays) / "neuron_d_nodes.json").read_text())
    nd_held = np.load(Path(args.arrays) / "neuron_d_held.npy", mmap_mode="r")
    nd_fit = np.load(Path(args.arrays) / "neuron_d_fit.npy", mmap_mode="r")
    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else []
    ck = lab["checkpoint"]
    rows, maps = {}, {}
    unit_by_topic = {}
    for u in plan["units"]:
        if u["kind"] in ("cluster", "subcluster", "puzzle"):
            for t in u.get("labelled_topics", []):
                unit_by_topic.setdefault(t, []).append(u)
    lab_by_unit = {}
    for c in lab["clusters"]:
        lab_by_unit[c["id"]] = c
        for s in c.get("subclusters", {}).get("items", []):
            lab_by_unit[s["id"]] = s
    core_units = {u["topic"]: u for u in plan["units"] if u["kind"] == "core_map"}
    lb_path = Path(args.labels) / "lang_breadth.json"
    lang_breadth = json.loads(lb_path.read_text())["units"] if lb_path.exists() else {}
    for x in node_ids:
        k = node_ids.index(x)
        row = {"name": tax.nodes[x]["name"], "contrast": lab["contrast"].get(x), "units": [], "boxes": [],
               "hub_align": None}   # [{run, schema, candidate, donor, host, warp}], problem-scoped (see DESIGN)
        passing_units, failed_units, unmeasured = [], [], []
        for u in unit_by_topic.get(x, []):
            pairs = U[u["key"]]
            g = (next(l["gates"] for l in lab_by_unit[u["unit"]]["labels"] if l["topic"] == x)
                 if u["unit"] in lab_by_unit else {"contrast": "propagated by co-activation (puzzle stage)"})
            ok, why = functional_pass(fres.get(u["unit"]), x, args.min_effect)
            fr = fres.get(u["unit"], {}).get("topics", {})
            ent = {"unit": u["unit"], "kind": u["kind"], "neurons": int(len(pairs)),
                   "selectivity": {kk: g.get(kk) for kk in ("d_fit", "d_held", "auc_held", "twin95_fit", "twin95_held",
                                                            "perm_z", "perm_p_bonf", "perm_p_emp_textlevel", "groups_fit", "contrast")},
                   "selectivity_global": {kk: g.get("global", {}).get(kk) for kk in ("d_fit", "d_held")} if g.get("global") else None,
                   "functional": {"topic": fr.get(x), "control": next((v for kk, v in fr.items() if kk.startswith("control(")), None),
                                  "pass": ok, "why_not": why},
                   "neuron_map": {"npz_key": f"{x}|{u['unit']}", "per_layer": per_layer(pairs)}}
            if u["unit"] in lang_breadth:          # a language label may really be a family or a rare-script unit
                ent["lang_breadth"] = lang_breadth[u["unit"]]
            maps[f"{x}|{u['unit']}"] = pairs.astype(np.int32)
            row["units"].append(ent)
            (passing_units if ok else unmeasured if ok is None else failed_units).append(u["unit"])
        for b in boxes:
            if b["topic"] == x:
                row["boxes"].append(b)
        # core tag: exactly the plan's core map for x (core neurons selected on FIT, chance-controlled), so the
        # functional result below measures the same neurons that the tag lists
        cu = core_units.get(x)
        df, dh = np.asarray(nd_fit[k], np.float32), np.asarray(nd_held[k], np.float32)
        row["selective_neurons"] = {"core_fit": int((core & (df >= args.core_d)).sum()),
                                    "pool_fit": int((~core & (df >= args.core_d)).sum()),
                                    "core_chance": cu.get("n_chance") if cu else None, "threshold_d": args.core_d}
        subsumed = sorted(t for t, us in unit_by_topic.items() if t != x and x in tax.ancestors(t)
                          and any(functional_pass(fres.get(uu["unit"]), t, args.min_effect)[0] for uu in us))
        if cu and cu.get("key"):
            pairs = U[cu["key"]]
            maps[f"{x}|core"] = pairs.astype(np.int32)
            dsel = dh[pairs[:, 0], pairs[:, 1]]
            ov = None
            eng = node_ids.index("lang.eng") if "lang.eng" in node_ids else None
            if eng is not None and x != "lang.eng":
                de = np.asarray(nd_fit[eng], np.float32)[pairs[:, 0], pairs[:, 1]]
                ov = round(float((de >= args.core_d).mean()), 3)
            ok, why = functional_pass(fres.get(cu["unit"]), x, args.min_effect)
            fr = fres.get(cu["unit"], {}).get("topics", {})
            row["core_tag"] = {
                "neuron_map": {"npz_key": f"{x}|core", "n": int(len(pairs)), "n_selective_fit": cu["n_selective"],
                               "per_layer": per_layer(pairs, dsel)},
                "selection": cu.get("selection"),
                "selectivity": {"d_fit_mean": round(float(df[pairs[:, 0], pairs[:, 1]].mean()), 3),
                                "d_held_mean": round(float(dsel.mean()), 3), "held_confirm_frac": cu.get("held_confirm_frac"),
                                "chance_count": cu.get("n_chance")},
                "english_overlap": ov,
                "functional": {"topic": fr.get(x), "control": next((v for kk, v in fr.items() if kk.startswith("control(")), None),
                               "pass": ok, "why_not": why}}
        if passing_units:
            row["status"] = "cluster"
        elif row["boxes"]:
            row["status"] = "box"
        elif unmeasured:                                  # an unmeasured unit may still pass: stays unverified
            row["status"] = "cluster_unverified"
            row["reason_not_split"] = "not_measured"
            row["failed_units"] = failed_units
        elif subsumed:
            row["status"] = "subsumed"
            row["reason_not_split"] = f"subsumed_by:{subsumed[0]}"
            row["subsumed_by"] = subsumed
        elif row.get("core_tag") or failed_units:
            row["status"] = "core_resident"
            ct = row.get("core_tag")
            if failed_units:
                reason = "cluster_failed_function"
            elif ct and ct["english_overlap"] is not None and ct["english_overlap"] >= args.english_overlap:
                reason = "overlaps_english_core"
            elif ct and ct["neuron_map"]["n"] < args.size_floor:
                reason = "below_size_floor"
            else:
                reason = "no_pool_cluster"         # selective neurons are in the core; no pool cluster carries x
            row["reason_not_split"] = reason
            row["failed_units"] = failed_units
        else:
            row["status"] = "absent"
            row["reason_not_split"] = ("selective core neurons at chance level" if cu and not cu.get("key")
                                       else "no selective neurons and no unit")
            if cu and not cu.get("key"):
                row["core_check"] = {k2: cu.get(k2) for k2 in ("n_selective", "n_chance", "held_confirm_frac", "note")}
        if row["status"] == "core_resident" and args.core_corpus and "repo_markdown" in args.core_corpus \
                and x.split(".")[0] in args.by_construction.split(","):
            row["core_carving_corpus"] = "includes repo_markdown"
            row["caveat"] = ("the core was carved to keep repo-style text cheap, so this topic's core residency is partly by "
                             "construction; it is not evidence of where the model naturally stores it (a prose-only core "
                             "would be needed to see it as a separable box)")
        rows[x] = row
    # roll-ups: a member topic moves into its parent-level merged unit when it holds its own score there
    for u in plan["units"]:
        if u["kind"] != "rollup":
            continue
        fr = fres.get(u["unit"])
        if not fr:
            for t in u["rollup_members"]:
                if t in rows:
                    rows[t]["rollup_candidate"] = {"unit": u["unit"], "status": "not measured"}
            continue
        ctl = next((v for k2, v in fr["topics"].items() if k2.startswith("control(")), None)
        ctl_ok = (ctl is not None and "add_over_twin" in ctl and ctl["add_over_twin"] >= -args.rollup_tol
                  and ctl.get("remove_over_twin", 1.0) <= args.rollup_tol)
        verdicts = {}
        for t in u["rollup_members"]:
            merged = fr["topics"].get(t, {}).get("add_over_twin")
            solo_units = [m for m in u["member_units"] if m in fres and t in fres[m]["topics"]]
            solo = max([fres[m]["topics"][t].get("add_over_twin") for m in solo_units
                        if fres[m]["topics"][t].get("add_over_twin") is not None], default=None)
            member_pass = any(functional_pass(fres[m], t, args.min_effect)[0] for m in solo_units)
            if merged is None or solo is None or ctl is None:
                verdicts[t] = {"merged_add_over_twin": merged, "standalone_add_over_twin": solo, "holds": None,
                               "why": "not measured (member skipped or no control)"}
                continue
            ok = merged >= solo - args.rollup_tol and ctl_ok and member_pass
            verdicts[t] = {"merged_add_over_twin": merged, "standalone_add_over_twin": solo, "holds": bool(ok),
                           "member_unit_passes": member_pass}
        measured = all(v["holds"] is not None for v in verdicts.values())
        all_ok = measured and all(v["holds"] for v in verdicts.values())
        for t, v in verdicts.items():
            if t not in rows:
                continue
            rows[t]["rollup"] = {"unit": u["unit"], "members": u["rollup_members"], "verdict": v, "control": ctl,
                                 "no_interference": all_ok if measured else "not measured"}
            if all_ok:
                rows[t]["status"] = "box"
                rows[t]["in_box"] = u["unit"]
                rows[t]["box_kind"] = "rollup: uncarved union of the members' units (gate/router shared at the parent level)"
    for b in boxes:
        if b["topic"] not in rows:
            rows[b["topic"]] = {"name": tax.nodes[b["topic"]]["name"], "status": "box", "units": [], "boxes": [b],
                                "note": "topic not in the probe corpus's tested nodes"}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sl = slug(ck)
    npz = out_dir / f"{sl}.topics.npz"
    np.savez_compressed(npz, **maps)
    counts = {}
    for r in rows.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    capmeta = Path(lab["capture"]) / "meta.json"
    fp_full = json.loads(capmeta.read_text()).get("fingerprint") if capmeta.exists() else None
    doc = {"schema": SCHEMA, "model": {"checkpoint": ck, "fingerprint": fp_full, "layers": L, "neurons_per_layer": N,
                                       "core_fraction": round(float(core.mean()), 4),
                                       "core_carving_corpus": args.core_corpus.split(",") if args.core_corpus else None},
           "taxonomy_version": tax.version, "mlx2_source_revision": T.git_rev(),
           "inputs": {"labels": str(args.labels), "labels_revision": lab["mlx2_source_revision"], "plan": str(args.plan),
                      "function": str(args.function) or None, "function_revision": fn.get("mlx2_source_revision"),
                      "capture": lab["capture"], "corpus_sha256": lab["corpus_sha256"], "clusters": lab["clusters_source"],
                      "core": plan["core"]},
           "gates": {"selectivity": lab["gates"], "functional": f"remove_over_twin >= {args.min_effect} and >= 2x |control|",
                     "core_tag": f"core neurons with d >= {args.core_d} on fit and held"},
           "neuron_maps": {"path": str(npz.relative_to(T.ROOT)) if npz.is_relative_to(T.ROOT) else str(npz),
                           "sha256": T.sha_file(npz), "key": "<topic>|<unit or 'core'>", "dtype": "int32 [n, 2] (layer, neuron)"},
           "counts": counts,
           "untested": {"taxonomy_leaves": sum(1 for n in tax.nodes.values() if n["kind"] == "leaf"),
                        "tested_nodes": len(node_ids),
                        "note": "leaves without probe texts are untested, not absent"},
           "topics": dict(sorted(rows.items())),
           "clusters": {c["id"]: {"neurons": c["neurons"], "status": c["status"], "labels": [l["topic"] for l in c["labels"]],
                                  "subclusters": [{"id": s["id"], "neurons": s["neurons"], "labels": [l["topic"] for l in s["labels"]]}
                                                  for s in c.get("subclusters", {}).get("items", [])],
                                  "functional": fres.get(c["id"], {}).get("topics")}
                        for c in lab["clusters"]}}
    T.write_json(out_dir / f"{sl}.topics.json", doc)
    print(f"wrote {out_dir / (sl + '.topics.json')}: {counts}")


if __name__ == "__main__":
    main()
