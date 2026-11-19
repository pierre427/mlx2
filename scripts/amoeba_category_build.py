#!/usr/bin/env python3
"""Category boxes + probe windows for the 2026-11-18 category corpora (CPU only).

Box spec (Pierre 2026-11-16, unchanged): >= 2 natural registers about half each; >= 1,500 fit docs of >= 256 tokens
per register; 2k-10k docs per box; val/held/held2; no AI; no medical text outside sci.medicine.
Split: sha256(group) % 8 -> 0-4 fit, 5 val, 6 held, 7 held2.

Inputs: candidate docs (amoeba_category_sources.py, amoeba_agent_sources.py) and the log-odds assignment
(amoeba_category_mine.py score). A subject doc counts for its assigned node and every ancestor; rule-labelled
agent/dialog docs count for their rule node. Docs are exclusive among siblings (one assigned node each).

Per box: registers with >= --reg-min docs are balanced to the smallest (capped at --reg-cap each, and
10,000 / registers); else the two largest, balanced. At most --per-group docs from one group (book, paper,
trajectory). Subject boxes use origin == natural only. agent.* / dialog.* boxes take model-generated docs (they
are the subject) and record the origin mix; "natural" there means the box has >= 2 source families.

Probes: per node, --probe-fit windows from fit docs and --probe-held from held/held2 docs, each the ~--window char
span of its doc densest in the node's terms (whitespace-snapped). Same row shape as topic-probes-9a1f35475.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amoeba_category_mine as MINE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = ("Pierre 2026-11-16: >=2 natural registers ~half each; >=1,500 fit docs >=256 tokens per register; "
        "2k-10k docs per box; val/held/held2; no AI; no medical outside sci.medicine")


def split_of(group):
    k = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 8
    return "fit" if k < 5 else ("val", "held", "held2")[k - 5]


def rev():
    return subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def behaviour(node):
    return node.split(".")[0] in ("agent", "dialog")


def best_window(text, terms, width):
    """Whitespace-snapped `width`-char window with the highest z-weighted term density."""
    if len(text) <= width:
        return text
    best, bs = 0.0, 0
    step = max(64, width // 4)
    for s in range(0, len(text) - width // 2, step):
        w = text[s:s + width]
        g, _ = MINE.grams(w)
        v = sum(terms.get(x, 0.0) for x in g)
        if v > best:
            best, bs = v, s
    s = bs
    if s:
        m = re.search(r"\s", text[s:s + 80])
        s += m.end() if m else 0
    e = s + width
    m = re.search(r"\s", text[e:e + 80])
    e += m.start() if m else 0
    return text[s:e].strip()


def random_window(d, width):
    """Deterministic random-offset window (offset from sha256(group)), whitespace-snapped: unbiased held text for
    nats/byte tests (the cross-model transport session, 2026-11-18), beside the topic-dense window."""
    text = d["text"]
    if len(text) <= width:
        return text
    s = int(hashlib.sha256(("rw" + d["group"]).encode()).hexdigest()[:8], 16) % (len(text) - width)
    m = re.search(r"\s", text[s:s + 80])
    s += m.end() if m else 0
    e = s + width
    m = re.search(r"\s", text[e:e + 80])
    e += m.start() if m else 0
    return text[s:e].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--assign", required=True, help="mine work dir (assign.jsonl, terms/)")
    ap.add_argument("--out-boxes", required=True)
    ap.add_argument("--out-probes", required=True)
    ap.add_argument("--taxonomy", default=str(ROOT / "docs/research/amoeba-topics/taxonomy.v1.json"))
    ap.add_argument("--reg-min", type=int, default=2400)
    ap.add_argument("--reg-cap", type=int, default=3400)
    ap.add_argument("--fit-target", type=int, default=1550, help="fit docs per register to aim for (spec: >= 1,500)")
    ap.add_argument("--per-group", type=int, default=3)
    ap.add_argument("--max-registers", type=int, default=3)
    ap.add_argument("--guard", nargs="*", default=[], help="guard shingle-hash files: docs sharing a 13-word shingle with an eval set are dropped")
    ap.add_argument("--per-group-code", type=int, default=40, help="code groups are repo/top-level dir")
    ap.add_argument("--window", type=int, default=520)
    ap.add_argument("--probe-fit", type=int, default=64)
    ap.add_argument("--probe-held", type=int, default=32)
    a = ap.parse_args()
    for d in (a.out_boxes, a.out_probes):
        p = Path(d)
        if p.exists() and any(p.iterdir()):
            raise SystemExit(f"refusing to overwrite {p}")
        p.mkdir(parents=True, exist_ok=True)
    nodes, kids = MINE.load_tax(a.taxonomy)
    W = Path(a.assign)
    # every doc is keyed by its global line position across --cands (the order the miner scored them), never by
    # its id: ids can collide across windows of one source
    assign = {r["ln"]: r for r in map(json.loads, open(W / "assign.jsonl"))}
    med_rule = {int(x) for x in open(W / "medical_rule_docs.txt")} if (W / "medical_rule_docs.txt").exists() else set()
    terms = {}
    for f in (W / "terms").glob("*.tsv"):
        terms[f.stem] = {ln.split("\t")[0]: float(ln.split("\t")[1]) for i, ln in enumerate(open(f)) if i}
    group_cap = {"code": a.per_group_code}
    # ---- pass 1: metadata only (no text kept), exact-text dedup across sources (first source wins)
    pools = defaultdict(lambda: defaultdict(list))            # node -> register -> [doc id]
    docs, seen, dropped = {}, set(), Counter()
    guard = set()
    for gf in a.guard:                            # contamination guards (eval sets): 13-word shingle hashes
        guard.update(x.strip() for x in open(gf) if x.strip())

    def contaminated(text):
        w = re.findall(r"\w+", text.lower())
        return any(hashlib.blake2b(" ".join(w[k:k + 13]).encode(), digest_size=8).hexdigest() in guard
                   for k in range(0, max(1, len(w) - 12), 4))
    ln = -1
    for p in a.cands:
        for line in open(p):
            ln += 1
            r = json.loads(line)
            lab = r.get("label") or {}
            if lab.get("method", "").startswith("rule:"):
                node, label = r["primary"], lab
                if ln in med_rule:
                    dropped["medical (rule-labelled doc)"] += 1
                    continue
            else:
                x = assign.get(ln)
                if not x:
                    continue
                node = x["node"]
                label = {"method": "logodds", "density": x["density"], "margin": x["margin"], "runner_up": x["runner_up"],
                         "source_label": r.get("primary") or r["cands"]}
            if node not in nodes or r["register"] == "sentinel":
                continue                      # sentinel = at-spec subject-boxes-99a59a894 fit docs: terms/competition only
            if node == MINE.MEDICAL or node.startswith(MINE.MEDICAL + "."):
                dropped["medical node"] += 1
                continue
            if behaviour(node) and label["method"] == "logodds":
                dropped["open doc on behaviour node"] += 1
                continue                      # agent.*/dialog.* accept only rule-labelled transcripts
            if not behaviour(node) and r.get("origin", "natural") != "natural":
                continue
            if guard and contaminated(r["text"]):
                dropped["eval-guard overlap (13-word shingle)"] += 1
                continue
            key = hashlib.sha256(re.sub(r"\s+", " ", r["text"]).strip().lower().encode()).hexdigest()
            if key in seen:
                dropped["exact duplicate text"] += 1
                continue
            seen.add(key)
            r.pop("text")
            r["label"], r["node"] = label, node
            docs[ln] = r
            for anc in MINE.ancestors(node):
                if anc in nodes:
                    pools[anc][r["register"]].append(ln)
    del seen
    print(f"{len(docs)} assigned docs over {len(pools)} nodes; dropped {dict(dropped)}", flush=True)
    rows_by_doc, boxes = defaultdict(list), {}
    order = lambda i: hashlib.sha256(docs[i]["group"].encode()).hexdigest()  # noqa: E731
    for b in sorted(pools):
        regs = {}
        for reg, ids in pools[b].items():
            ids = sorted(set(ids), key=order)
            per, keep, gc = Counter(), [], group_cap.get(reg, a.per_group)
            for i in ids:
                g = docs[i]["group"]
                if per[g] < gc:
                    per[g] += 1
                    keep.append(i)
            regs[reg] = keep
        big = {k: v for k, v in regs.items() if len(v) >= a.reg_min}
        # at most --max-registers (3) registers: more would push the per-register cap (10,000 / registers) below
        # the >= 1,500 fit docs each register needs; the largest qualifying registers are kept
        big = dict(sorted(big.items(), key=lambda kv: -len(kv[1]))[: a.max_registers])
        use = big if len(big) >= 2 else dict(sorted(regs.items(), key=lambda kv: -len(kv[1]))[:2])
        cap = min(a.reg_cap, 10000 // max(1, len(use)))
        # equal size per register (about half each): the smallest count at which EVERY register has >= --fit-target
        # fit docs, bounded by the cap and the smallest register (the split share varies around 5/8)
        def need(v):
            f = 0
            for k, i in enumerate(v, 1):
                f += split_of(docs[i]["group"]) == "fit"
                if f >= a.fit_target:
                    return k
            return len(v)
        n = min(cap, min(len(v) for v in use.values()), max(need(v) for v in use.values())) if use else 0
        info = {"registers": {}, "two_registers": len(use) >= 2,
                "dropped_registers": {k: len(v) for k, v in regs.items() if k not in use},
                "behaviour_box": behaviour(b)}
        for reg, v in use.items():
            c, src, lic, org = Counter(), Counter(), Counter(), Counter()
            for i in v[:n]:
                d = docs[i]
                sp = split_of(d["group"])
                c[sp] += 1
                src[d["source"].split(":")[0]] += 1
                lic[d["license"]] += 1
                org[d.get("origin", "natural")] += 1
                rows_by_doc[i].append((b, reg, sp))
            dates = sorted(str(docs[i].get("created") or "")[:10] for i in v[:n] if re.match(r"\d{4}-\d\d", str(docs[i].get("created") or "")))
            info["registers"][reg] = {"available": len(v), "taken": min(n, len(v)), **dict(c),
                                      "created_range": [dates[0], dates[-1]] if dates else None,
                                      "created_on_or_after_2022-12-01": sum(d >= "2022-12-01" for d in dates),
                                      "undated": min(n, len(v)) - len(dates),
                                      "sources": dict(src.most_common(8)), "licenses": dict(lic.most_common(6)),
                                      "origins": dict(org)}
        tot = sum(r["taken"] for r in info["registers"].values())
        fit_min = min((r.get("fit", 0) for r in info["registers"].values()), default=0)
        ok = info["two_registers"] and fit_min >= 1500 and 2000 <= tot <= 10000
        if not ok:
            why = []
            if not info["two_registers"]:
                why.append("one register only")
            if fit_min < 1500:
                why.append(f"smaller register has {fit_min} fit docs (< 1,500)")
            if tot < 2000:
                why.append(f"{tot} docs (< 2,000)")
            info["short_reason"] = "; ".join(why)
        info.update({"docs": tot, "meets_spec": ok})
        boxes[b] = info
    # probe selection (ids only): one doc per group per node, registers round-robin
    probe_sel = defaultdict(list)                              # doc id -> [(node, split)]
    ptop = {}
    for b in sorted(pools):
        ids = sorted({i for v in pools[b].values() for i in v}, key=order)
        fit = [i for i in ids if split_of(docs[i]["group"]) == "fit"]
        held = [i for i in ids if split_of(docs[i]["group"]) in ("held", "held2")]

        def take(pool, k):
            out, seen_g = [], set()
            byreg = defaultdict(list)
            for i in pool:
                byreg[docs[i]["register"]].append(i)
            its = [iter(v) for v in byreg.values()]
            while len(out) < k and its:
                for it in list(its):
                    i = next(it, None)
                    if i is None:
                        its.remove(it)
                        continue
                    if docs[i]["group"] in seen_g:
                        continue
                    seen_g.add(docs[i]["group"])
                    out.append(i)
                    if len(out) >= k:
                        break
            return out
        sel = [("fit", i) for i in take(fit, a.probe_fit)] + [("held", i) for i in take(held, a.probe_held)]
        for sp, i in sel:
            probe_sel[i].append((b, sp))
        c = Counter(sp for sp, _ in sel)
        ptop[b] = {"fit": c["fit"], "held": c["held"], "tested": c["fit"] >= 16 and c["held"] >= 16,
                   "registers": dict(Counter(docs[i]["register"] for _, i in sel)),
                   "origins": dict(Counter(docs[i].get("origin", "natural") for _, i in sel))}
    R = rev()
    out, po = Path(a.out_boxes), Path(a.out_probes)
    # ---- pass 2: stream texts once, writing box rows and probe windows for the selected docs
    nrow = nprobe = 0
    with gzip.open(out / "corpus.jsonl.gz", "wt") as fb, open(po / "corpus.jsonl", "w") as fp:
        i = -1
        for p in a.cands:
            for line in open(p):
                i += 1
                if i not in docs or (i not in rows_by_doc and i not in probe_sel):
                    continue
                r = json.loads(line)
                d, text = docs[i], r["text"]
                for b, reg, sp in rows_by_doc.get(i, ()):
                    fb.write(json.dumps({"id": f"cb{nrow:07d}", "box": b, "node": d["node"], "register": reg, "split": sp,
                                         "text": text, "origin": d.get("origin", "natural"), "source": d["source"],
                                         "source_revision": d["source_revision"], "group": d["group"],
                                         "license": d["license"], "redistribute": d.get("redistribute", ""),
                                         "created": d.get("created", ""), "label": d["label"],
                                         "sha256": hashlib.sha256(text.encode()).hexdigest()}, ensure_ascii=False) + "\n")
                    nrow += 1
                for b, sp in probe_sel.get(i, ()):
                    dd = {**d, "text": text}
                    for kind, w in (("dense", best_window(text, terms.get(b) or {}, a.window)),
                                    ("random", random_window(dd, a.window))):
                        fp.write(json.dumps({"id": f"cp{nprobe:06d}", "topics": [b], "split": sp, "window_kind": kind,
                                             "text": w, "source": d["source"], "origin": d.get("origin", "natural"),
                                             "group": d["group"], "license": d["license"],
                                             "redistribute": d.get("redistribute", ""), "method": d["label"]["method"],
                                             "register": d["register"], "node": d["node"], "created": d.get("created", ""),
                                             "source_revision": d["source_revision"],
                                             "sha256": hashlib.sha256(w.encode()).hexdigest()}, ensure_ascii=False) + "\n")
                        nprobe += 1
    (out / "terms").mkdir(exist_ok=True)
    for f in (W / "terms").glob("*"):
        (out / "terms" / f.name).write_bytes(f.read_bytes())
    meets = sorted(b for b, i in boxes.items() if i["meets_spec"])
    man = {"schema": "mlx2.amoeba-category-boxes.v1", "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "builder": "scripts/amoeba_category_build.py", "mlx2_source_revision": R, "spec": SPEC,
           "split": "sha256(group) % 8: 0-4 fit, 5 val, 6 held, 7 held2",
           "date_cut": "none (Pierre 2026-11-18: probe corpora, not training). Every row keeps `created`; per-register "
                       "created_range and the count on/after 2022-12-01 are recorded, so a pre-2022-12 view is a filter",
           "register_rule": f"registers with >= {a.reg_min} docs; equal take per register = smallest count giving every "
                            f"register >= {a.fit_target} fit docs, capped at {a.reg_cap} "
                            f"(and 10,000 / registers), at most {a.max_registers} (the largest); else the two largest, balanced; <= "
                            f"{a.per_group} docs per group per box ({a.per_group_code} for code, grouped by repo/top dir)",
           "membership": "a doc counts for its assigned node and every ancestor; siblings are exclusive",
           "dedup": "exact full text (whitespace-normalised, lower-cased) across all sources, first source wins",
           "medical": "docs assigned to sci.medicine are dropped; medicine is a rival for every node in scoring",
           "behaviour_boxes": "agent.* / dialog.*: rule-labelled transcripts only; model-generated text is the subject; "
                              "origin recorded per row and per register; rendered as plain 'System:/User:/Assistant:/Tool:' "
                              "transcripts (re-render in the target model's chat template at probe time if needed)",
           "corpus": "corpus.jsonl.gz", "corpus_sha256": hashlib.sha256((out / "corpus.jsonl.gz").read_bytes()).hexdigest(),
           "rows": nrow, "unique_docs": len(rows_by_doc),
           "assignment": json.loads((W / "assign_meta.json").read_text()) if (W / "assign_meta.json").exists() else {},
           "dropped_in_build": dict(dropped),
           "terms": "terms/<node>.tsv (term, z, count, docs) + terms/_terms.json",
           "inputs": {"cands": a.cands, "assign": str(W)},
           "boxes": boxes, "meets_spec": meets, "short": sorted(b for b in boxes if b not in meets)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(f"boxes: {len(boxes)}; meet spec {len(meets)}; rows {nrow}", flush=True)
    pman = {"schema": "mlx2.amoeba-topic-probes.v1", "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "builder": "scripts/amoeba_category_build.py", "mlx2_source_revision": R,
            "taxonomy_version": json.load(open(a.taxonomy))["version"], "corpus_path": "corpus.jsonl",
            "corpus_sha256": hashlib.sha256((po / "corpus.jsonl").read_bytes()).hexdigest(), "items": nprobe,
            "per_split_quota": {"fit": a.probe_fit, "held": a.probe_held},
            "rules": {"split": "probe fit from box fit docs; probe held from box held/held2 docs (sha256(group) % 8)",
                      "window_chars": a.window,
                      "window": "two rows per selected doc: window_kind=dense (densest span of the node's log-odds terms; "
                                "terms are mined from fit docs only) and window_kind=random (offset from sha256(group)); "
                                "both whitespace-snapped",
                      "selection": "one doc per group per node; registers round-robin"},
            "companion_boxes": str(out), "topics": ptop,
            "summary": {"topics_with_items": len(ptop), "tested": sum(v["tested"] for v in ptop.values())}}
    (po / "manifest.json").write_text(json.dumps(pman, indent=1, ensure_ascii=False) + "\n")
    print(f"probes: {nprobe} windows over {len(ptop)} topics ({pman['summary']['tested']} tested)", flush=True)

if __name__ == "__main__":
    main()
