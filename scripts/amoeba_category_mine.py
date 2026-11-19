#!/usr/bin/env python3
"""Distinctive-term mining and assignment for the 2026-11-18 category corpora (Pierre 2026-11-18: "focus on phrases
and terms that are unique to each subject matter"). CPU only.

terms   For every taxonomy node with source-labelled candidates, count unigrams + bigrams over up to --term-docs
        source-confident docs (primary == node, or a single candidate; a node also pools its descendants) and
        compare with a general-English background (enwiki 20231101 + FineWeb-Edu, local) by weighted log-odds
        with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008). The top --top terms by z (count
        >= 10, in >= 5 docs) form the node's term list: terms/<node>.tsv (term, z, count, docs).
score   Each candidate doc gets, for every node in its candidate subtrees (a nominated node and all its
        descendants; '*' = every node with a term list), density = sum of z over its term hits (<= 3 per term)
        per 100 tokens. The doc is assigned to the best-scoring node if its density >= the node's threshold
        (25th percentile of its source-confident docs) and its margin over the best node outside its own
        ancestor line is >= --margin (x --open-margin for '*' docs). Writes assign.jsonl:
        {id, node, density, margin, runner_up, method: "logodds"}.

The term lists are the record of each subject's unique vocabulary; they are committed with the builds.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amoeba_category_map import SE_SITE  # noqa: E402
TOK = re.compile(r"\\[a-z]+|[a-z_][a-z0-9_'+#]*|[^\sa-z0-9]{2,3}")
WORD = re.compile(r"^[a-z_\\]")
STOP = set("""the of and to in a is that for it as was with be by on not he i this are or his from at which but have an
they you were her she there had been one we all their has would when if more no its so can out what about who will
them some into my may also than then only other these could do such any our like time two first very up over most
new even after just me him how those through where much well should us before because each your many own same
between both under while however another during against without within upon around since among off whether being
yet still often""".split())


# symbol-run tokens are subject signal only where syntax is the subject (code, tool calls, harness text); in prose
# subjects they mark the source format (LaTeX delimiters), not the subject
PROSE = re.compile(r"^(?:\\[a-z]+|[a-z_][a-z0-9_'+#]*)$")
ROLE_WORDS = {"user", "assistant", "system", "tool", "tools", "human"}
SYMBOL_ROOTS = {"prog", "swe", "agent", "dialog", "sec", "cs"}
PUNCT = re.compile(r"^[.,!?;:\"'\u2018\u2019\u201c\u201d\-\u2013\u2014()\[\]*/]+$")


def toks(text):
    return [w for w in TOK.findall(text.lower()) if (len(w) > 1 or not WORD.match(w)) and not PUNCT.match(w)]


def grams(text, limit=1500):
    """Unigrams (minus stop words) + bigrams of ADJACENT word tokens (a symbol token between two words breaks the
    pair, exactly as density() pairs them)."""
    t = toks(text)[:limit]
    out = [w for w in t if w not in STOP]
    out += [f"{a} {b}" for a, b in zip(t, t[1:]) if WORD.match(a) and WORD.match(b) and a not in STOP and b not in STOP]
    return out, len(t)


def split_of(group):
    k = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 8
    return "fit" if k < 5 else ("val", "held", "held2")[k - 5]


MEDICAL = "sci.medicine"                    # box spec: no medical text outside sci.medicine


def load_tax(path):
    nodes = {n["id"]: n for n in json.load(open(path))["nodes"] if not n["id"].startswith("lang")}
    kids = defaultdict(list)
    for n in nodes.values():
        if n.get("parent") in nodes:
            kids[n["parent"]].append(n["id"])
    return nodes, kids


def ancestors(node):
    p = node.split(".")
    return {".".join(p[:i]) for i in range(1, len(p) + 1)}


def subtree(node, kids):
    out, st = [], [node]
    while st:
        x = st.pop()
        out.append(x)
        st.extend(kids.get(x, []))
    return out


def confident(r, nodes):
    """The doc's source-confident node: its primary label, else the single deepest candidate once candidates that
    are ancestors of another candidate are dropped (SE: site default + one tag node -> the tag node)."""
    if r.get("primary") and r["primary"] in nodes:
        return r["primary"]
    c = [x for x in r.get("cands", []) if x in nodes]
    if r.get("site"):                       # StackExchange: tag nodes beat the site default, which is a fallback
        base = SE_SITE.get(r["site"])
        tags = [x for x in c if x != base]
        tdeep = [x for x in tags if not any(y != x and y.startswith(x + ".") for y in tags)]
        if len(tdeep) == 1:
            return tdeep[0]
        return base if not tags and base in nodes else None
    deep = [x for x in c if not any(y != x and y.startswith(x + ".") for y in c)]
    return deep[0] if len(deep) == 1 else None


def iter_cands(paths):
    for p in paths:
        for line in open(p):
            yield json.loads(line)


# ------------------------------------------------------------------ terms
def _count(args):
    node, texts = args
    c, df = Counter(), Counter()
    n = 0
    for t in texts:
        g, k = grams(t)
        n += k
        c.update(g)
        df.update(set(g))
    # keep counts >= 2 for the reference sums; candidate terms are filtered later (count >= 10, >= 5 docs)
    return node, {w: v for w, v in c.items() if v >= 2}, {w: df[w] for w, v in c.items() if v >= 2}, n


def do_terms(a):
    """Exclusive counts: each fit-split doc counts once, at its confident node. A node's own counts are the sum over
    its subtree; its reference is background + every other node's exclusive counts (siblings, cousins, other roots)
    = background + TOTAL - SUBTREE. Term docs per (node, register) are the lowest-hash fit docs (bounded memory),
    balanced round-robin across registers. Held/val docs never shape terms (they are probe/battery text)."""
    import heapq
    nodes, kids = load_tax(a.taxonomy)
    out = Path(a.work)
    (out / "terms").mkdir(parents=True, exist_ok=True)
    cap = a.term_docs
    heaps = defaultdict(lambda: defaultdict(list))        # node -> register -> max-heap of (-hash, text), size <= cap
    for r in iter_cands(a.cands):
        conf = confident(r, nodes)
        if not conf or split_of(r["group"]) != "fit":
            continue
        h = int(hashlib.sha256(r["group"].encode()).hexdigest()[8:16], 16)
        hp = heaps[conf][r["register"]]
        item = (-h, r["id"], r["text"][:4000])
        if len(hp) < cap:
            heapq.heappush(hp, item)
        elif item > hp[0]:
            heapq.heapreplace(hp, item)
    jobs = []
    for node, regs in heaps.items():
        lists = [sorted(v, reverse=True) for v in regs.values()]
        take, k = [], 0
        while len(take) < cap and any(k < len(v) for v in lists):
            take += [v[k][2] for v in lists if k < len(v)]
            k += 1
        jobs.append((node, take[:cap]))
    del heaps
    print(f"terms: {len(jobs)} nodes with exclusive confident fit docs", flush=True)
    with Pool(a.procs) as pool:
        counted = pool.map(_count, jobs, chunksize=1)
    excl = {node: (c, df, n) for node, c, df, n in counted}
    total, total_n = Counter(), 0
    for c, _, n in excl.values():
        total.update(c)
        total_n += n
    vocab = set(total)
    bg, bgn, nbg = Counter(), 0, 0
    for path in a.background:
        for i, line in enumerate(open(path)):
            if i % a.bg_stride:
                continue
            g, k = grams(json.loads(line)["text"][:4000])
            bg.update(w for w in g if w in vocab)
            bgn += k
            nbg += 1
            if nbg >= a.bg_docs:
                break
    print(f"background: {nbg} docs, {bgn} tokens, {len(bg)} shared keys", flush=True)
    a0 = a.prior
    summary = {}
    for node in sorted(nodes):
        sub = [x for x in subtree(node, kids) if x in excl]
        if not sub:
            continue
        c, df, n = Counter(), Counter(), 0
        for x in sub:
            c.update(excl[x][0])
            df.update(excl[x][1])
            n += excl[x][2]
        docs_n = sum(len(next(j for j in jobs if j[0] == x)[1]) for x in sub)
        if docs_n < a.min_docs:
            continue
        ref_n = bgn + total_n - n
        rows = []
        for w, y in c.items():
            if y < 10 or df[w] < 5:
                continue
            yb = bg.get(w, 0) + total.get(w, 0) - y          # every other node's exclusive count
            aw = a0 * (bg.get(w, 0) + total.get(w, 0) + 0.5) / (bgn + total_n + 0.5 * len(vocab))
            d = math.log((y + aw) / (n + a0 - y - aw)) - math.log((yb + aw) / (ref_n + a0 - yb - aw))
            z = d / math.sqrt(1 / (y + aw) + 1 / (yb + aw))
            rows.append((z, w, y, df[w]))
        rows.sort(reverse=True)
        sym_ok = node.split(".")[0] in SYMBOL_ROOTS
        if node.split(".")[0] in ("agent", "dialog"):             # transcript role labels are our rendering, not subject
            rows = [r for r in rows if not (set(r[1].split(" ")) & ROLE_WORDS)]
        top = [r for r in rows if r[0] > 0 and (sym_ok or all(PROSE.match(p) for p in r[1].split(" ")))][: a.top]
        if not top:
            continue
        with open(out / "terms" / f"{node}.tsv", "w") as f:
            f.write("term\tz\tcount\tdocs\n")
            for z, w, y, d in top:
                f.write(f"{w}\t{z:.2f}\t{y}\t{d}\n")
        summary[node] = {"docs": docs_n, "tokens": n, "terms": len(top), "subtree_nodes_with_docs": len(sub),
                         "top12": [w for _, w, _, _ in top[:12]]}
    meta = {"method": "weighted log-odds, informative Dirichlet prior (Monroe et al. 2008); node subtree counts vs "
                      "(general background + every other node's exclusive counts)",
            "prior_a0": a0, "background": {"files": a.background, "docs": nbg, "tokens": bgn, "stride": a.bg_stride},
            "term_docs_cap": cap, "term_doc_sampling": "fit-split docs only, counted once at their confident node; "
                                                       "lowest-hash per (node, register), round-robin across registers",
            "min_count": 10, "min_docs": 5, "top": a.top, "tokenizer": TOK.pattern, "nodes": summary}
    json.dump(meta, open(out / "terms" / "_terms.json", "w"), indent=1)
    print(f"wrote {len(summary)} term lists", flush=True)


# ------------------------------------------------------------------ score
TERMS = {}


def _init(work, top):
    inv = defaultdict(list)
    for f in glob.glob(os.path.join(work, "terms", "*.tsv")):
        node = os.path.basename(f)[:-4]
        for i, line in enumerate(open(f)):
            if i == 0 or i > top:
                continue
            w, z, _, _ = line.rstrip("\n").split("\t")
            inv[w].append((node, float(z)))
    TERMS["inv"] = dict(inv)
    TERMS["bifirst"] = {w.split(" ")[0] for w in inv if " " in w}
    TERMS["nodes"] = {n for v in inv.values() for n, _ in v}


def density(text, limit=600):
    """z-weighted hits per 100 tokens for every node with a term hit; scoring reads the first `limit` tokens and
    builds a bigram only where its first word starts a listed bigram (same terms as grams(), much faster)."""
    t = toks(text)[:limit]
    inv, bif = TERMS["inv"], TERMS["bifirst"]
    c = Counter(w for w in t if w not in STOP)
    for a_, b_ in zip(t, t[1:]):
        if a_ in bif and WORD.match(a_) and WORD.match(b_) and b_ not in STOP:
            c[f"{a_} {b_}"] += 1
    s = defaultdict(float)
    for w, k in c.items():
        hit = inv.get(w)
        if hit:
            k = min(k, 3)
            for node, z in hit:
                s[node] += z * k
    n = max(len(t), 50)
    return {k: v * 100 / n for k, v in s.items()}


def _score(line):
    r = json.loads(line)
    d = density(r["text"])
    return {"id": r["id"], "cands": r["cands"], "primary": r.get("primary"), "group": r["group"], "site": r.get("site"),
            "d": {k: round(v, 2) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:40]}}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


def do_score(a):
    nodes, kids = load_tax(a.taxonomy)
    out = Path(a.work)
    scored = out / "scored.jsonl"
    with (Pool(a.procs, initializer=_init, initargs=(a.work, a.top)) if not a.reuse_scored else _Null()) as pool, \
            (open(scored, "w") if not a.reuse_scored else _Null()) as f:
        for p in ([] if a.reuse_scored else a.cands):
            n = 0
            for r in pool.imap(_score, open(p), chunksize=1024):
                f.write(json.dumps(r) + "\n")
                n += 1
            print(f"  scored {os.path.basename(p)}: {n}", flush=True)
    _init(a.work, a.top)
    have = TERMS["nodes"]
    # thresholds: q25 density on own node among source-confident docs
    own = defaultdict(list)
    for line in open(scored):
        r = json.loads(line)
        conf = confident(r, nodes)
        if conf and split_of(r["group"]) == "fit":
            for anc in ancestors(conf):              # inner nodes and roots get thresholds from their subtrees
                if anc in have:
                    own[anc].append(r["d"].get(anc, 0.0))
    thr = {}
    for k, v in own.items():
        v.sort()
        thr[k] = v[len(v) // 4] if len(v) >= 20 else (v[len(v) // 2] if v else 0.0)
    allnodes = sorted(have)
    n_ok, why, depth = 0, Counter(), Counter()
    med_drop = open(out / "medical_rule_docs.txt", "w")
    with open(out / "assign.jsonl", "w") as f:
        for ln, line in enumerate(open(scored)):
            r = json.loads(line)
            d = r["d"]
            open_ = "*" in r["cands"]
            if any(c.split(".")[0] in ("agent", "dialog") for c in r["cands"]):
                # rule-labelled behaviour docs are not assigned here, but they must not carry medical text either
                med = max((v for k, v in d.items() if k == MEDICAL or k.startswith(MEDICAL + ".")), default=0.0)
                other = max((v for k, v in d.items() if not (k == MEDICAL or k.startswith(MEDICAL + "."))), default=0.0)
                if med > 0 and (other <= 0 or other / med < a.margin):
                    med_drop.write(f"{ln}\n")
                    why["medical (rule-labelled doc)"] += 1
                continue
            # open ('*') docs never go to behaviour nodes: agent.*/dialog.* take only rule-labelled transcripts
            allowed = {x for x in allnodes if x.split(".")[0] not in ("agent", "dialog")} if open_ else \
                {x for c in r["cands"] if c in nodes for x in subtree(c, kids)} & have
            if not allowed:
                why["no term list for candidates"] += 1
                continue
            ranked = sorted(((v, k) for k, v in d.items() if k in allowed), reverse=True)
            if not ranked:
                why["no term hits"] += 1
                continue
            need = a.margin * (a.open_margin if open_ else 1.0)
            # top-down: enter at the best node's root (or the shallowest allowed node on its line), then descend
            # into the best child while it clears its own threshold and beats its siblings by the margin. Roots
            # would otherwise win every density race (a root's term list covers its whole subject).
            line_ = sorted((x for x in ancestors(ranked[0][1]) if x in allowed), key=lambda x: x.count("."))
            cur = line_[0]
            if cur == MEDICAL or cur.startswith(MEDICAL + "."):
                why["medical"] += 1
                continue
            med = max((v for k, v in d.items() if k == MEDICAL or k.startswith(MEDICAL + ".")), default=0.0)
            top_rivals = [(v, k) for k, v in d.items() if k.split(".")[0] != cur.split(".")[0] and
                          not (k == MEDICAL or k.startswith(MEDICAL + "."))] + [(med, MEDICAL)]
            rv, rival = max(top_rivals)
            # the entry node's subject density: best density anywhere on its subtree
            entry_v = max(v for k, v in d.items() if k == cur or k.startswith(cur + "."))
            margin = entry_v / rv if rv > 0 else 99.0
            med_margin = entry_v / med if med > 0 else 99.0
            if d.get(cur, 0.0) < thr.get(cur, 0.0):
                why["below node threshold"] += 1
                continue
            # labelled docs whose candidates span several subtrees: the entry subtree must beat the others too
            other_v = max((v for k, v in d.items() if k in allowed and k != cur and not k.startswith(cur + ".")
                           and not cur.startswith(k + ".")), default=0.0)
            if not open_ and other_v > 0 and entry_v / other_v < need:
                why["margin (between candidate subtrees)"] += 1
                continue
            # source-labelled docs (SE site/tags, arXiv category, peS2o field, shelf) already name their subject:
            # only medicine must be beaten at the root; unlabelled ('*') docs must beat every other subject too.
            # Exclusivity among siblings is the descent margin below in both cases.
            if med_margin < need:
                why["medical rival"] += 1
                continue
            if open_ and margin < need:
                why["margin (open doc vs other subjects)"] += 1
                continue
            sib_margin = None
            while True:
                ch = [c for c in kids.get(cur, []) if c in allowed and c != MEDICAL]
                if not ch:
                    break
                sc = sorted(((d.get(c, 0.0), c) for c in ch), reverse=True)
                bv, bc = sc[0]
                sv = sc[1][0] if len(sc) > 1 else 0.0
                sm = bv / sv if sv > 0 else 99.0
                if bv <= 0 or bv < thr.get(bc, 0.0) or sm < need:
                    break
                cur, sib_margin = bc, sm
            if cur == MEDICAL or cur.startswith(MEDICAL + "."):
                why["medical"] += 1
                continue
            depth[cur.count(".")] += 1
            f.write(json.dumps({"ln": ln, "id": r["id"], "node": cur, "density": round(d.get(cur, entry_v), 2),
                                "margin": round(margin, 3), "sibling_margin": sib_margin and round(sib_margin, 3),
                                "runner_up": rival, "threshold": round(thr.get(cur, 0.0), 2),
                                "method": "logodds"}) + "\n")
            n_ok += 1
    med_drop.close()
    json.dump({"thresholds": thr, "assigned": n_ok, "dropped": dict(why), "margin": a.margin,
               "open_margin": a.open_margin, "assigned_by_depth": dict(depth),
               "rule": "top-down: enter at the subject root (must beat medicine; unlabelled docs must also beat every other "
                       "subject), descend into the best child while it clears its threshold and beats its siblings by "
                       "the margin"}, open(out / "assign_meta.json", "w"), indent=1)
    print(f"assigned {n_ok}; dropped {dict(why)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["terms", "score"])
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--taxonomy", default=str(ROOT / "docs/research/amoeba-topics/taxonomy.v1.json"))
    ap.add_argument("--background", nargs="*", default=[])
    ap.add_argument("--bg-docs", type=int, default=60000)
    ap.add_argument("--bg-stride", type=int, default=97)
    ap.add_argument("--term-docs", type=int, default=2000)
    ap.add_argument("--min-docs", type=int, default=40)
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--prior", type=float, default=1000.0)
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--open-margin", type=float, default=1.5)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--reuse-scored", action="store_true", help="score: reuse work/scored.jsonl, redo assignment only")
    a = ap.parse_args()
    {"terms": do_terms, "score": do_score}[a.what](a)


if __name__ == "__main__":
    main()
