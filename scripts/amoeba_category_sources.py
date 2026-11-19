#!/usr/bin/env python3
"""Candidate extraction for the 2026-11-18 category corpora (CPU only; Pierre 2026-11-18, survey in amoeba-corpus
surveys/category-sources-20261118.md). Each subcommand reads one downloaded source and writes candidate docs as
jsonl, one doc per row:

  {id, group, register, text, source, source_revision, license, redistribute, origin, created, cands, primary}

`cands` are taxonomy nodes the source label nominates (amoeba_category_map.py); `primary` is the source's own
best label where it has one. Final assignment and filtering is by term log-odds (amoeba_category_mine.py).
No date cut by default (probe corpora, not training; Pierre 2026-11-18); --created-before YYYY-MM-DD sets one.
Medical labels are dropped (box spec).

Docs are 1,100-6,000 chars (>= ~256 tokens). Long works (papers, books) give one span per paper and up to
--book-chunks spans per book; every span of a work shares its group, so a work never spans splits.

Run with python -I. Downloads live in a local sources directory (resumable, see the survey).
"""
from __future__ import annotations

import argparse
import bz2
import csv
import glob
import gzip
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amoeba_category_map as M  # noqa: E402

MIN, MAX = 1100, 6000
TAG = re.compile(r"<[^>]+>")
PRE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S)
TECH_SITES = {"stackoverflow", "softwareengineering", "codereview", "unix", "superuser", "askubuntu", "serverfault",
              "devops", "dba", "security", "reverseengineering", "tex", "gamedev", "computergraphics", "arduino",
              "networkengineering", "crypto", "cs", "proofassistants", "scicomp"}
SA = "share-alike"


def clean_html(s, keep_code):
    s = s or ""
    if keep_code:
        s = PRE.sub(lambda m: "\n\n```\n" + html.unescape(TAG.sub("", m.group(1))).strip("\n") + "\n```\n\n", s)
    else:
        s = PRE.sub(" ", s)
    s = re.sub(r"</p>|<br\s*/?>|</li>|</h\d>|</blockquote>", "\n\n", s)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(TAG.sub("", s))).strip()


def span(text, start=0):
    """One 1,100-6,000 char span of whole paragraphs from `start`; '' if too short."""
    out, n = [], 0
    for p in re.split(r"\n\s*\n", text[start:]):
        p = p.strip()
        if len(p) < 40:
            continue
        if n + len(p) > MAX and n >= MIN:
            break
        out.append(p[:MAX])
        n += len(p) + 2
        if n >= MAX:
            break
    t = "\n\n".join(out)[:MAX]
    return t if len(t) >= MIN else ""


class Out:
    def __init__(self, path):
        self.f = open(path, "w")
        self.n = Counter()

    def put(self, **r):
        r.setdefault("origin", "natural")
        r["id"] = hashlib.sha256((r["source"] + "\0" + r["text"][:200]).encode()).hexdigest()[:20]
        r["cands"] = sorted(set(c for c in r.get("cands", []) if c))
        if not r["cands"]:
            return
        self.f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.n[r.get("primary") or r["cands"][0]] += 1

    def close(self, what):
        self.f.close()
        print(f"{what}: {sum(self.n.values())} docs; top: {self.n.most_common(12)}", flush=True)


# ---------------------------------------------------------------- StackExchange (archive.org 2024-04, with tags)
def do_se(a):
    o = Out(Path(a.out) / f"se{('-' + a.only) if a.only else ''}.jsonl")
    for z in sorted(Path(a.src).glob("*.7z")):
        site = z.name.split(".")[0].replace("-Posts", "")
        if (a.only and site not in a.only.split(",")) or site in M.SE_SKIP:
            continue
        base = M.SE_SITE.get(site)
        if not base:
            if not a.only:
                print(f"  skip {site}: not mapped (covered by subject-boxes-99a59a894 or off-target)", flush=True)
            continue
        if not z.with_suffix(".7z.done").exists() and not (Path(a.src) / (z.name + ".done")).exists() and site not in a.trust:
            print(f"  skip {site}: download not complete", flush=True)
            continue
        member = "Posts.xml"
        p = subprocess.Popen(["7z", "x", "-so", str(z), member], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        keep_code = site in TECH_SITES
        qs, per = {}, Counter()
        cap = a.per_cat if site != "stackoverflow" else a.so_per_cat
        root = None
        for ev, el in ET.iterparse(p.stdout, events=("start", "end")):
            if root is None:
                root = el                               # cleared below so Posts.xml rows do not pile up
            if ev != "end" or el.tag != "row":
                continue
            r = el.attrib
            if r.get("CreationDate", "") < M.CUTOFF and int(r.get("Score", "0")) >= 1:
                if r.get("PostTypeId") == "1":
                    tags = r.get("Tags", "")
                    tl = [t for t in re.split(r"[<>|]", tags) if t]
                    cands = set(M.tag_candidates(tags))
                    prim = None
                    if site == "stackoverflow":
                        langs = [M.SO_LANG[t] for t in tl if t in M.SO_LANG]
                        prim = langs[0] if langs else None
                        cands |= set(langs)
                        if not cands:
                            el.clear()
                            continue
                    else:
                        cands.add(base)
                    key = prim or (sorted(cands - {base}) or [base])[0]
                    if per[key] < cap:
                        per[key] += 1
                        qs[r["Id"]] = [r.get("Title", ""), r.get("Body", ""), r.get("CreationDate"), tags, cands, prim, []]
                elif r.get("PostTypeId") == "2":
                    q = qs.get(r.get("ParentId"))
                    if q is not None:
                        q[6].append((int(r.get("Score", "0")), r.get("Body", "")))
            el.clear()
            root.clear()
        p.wait()
        n0 = sum(o.n.values())
        lic = "CC BY-SA 4.0 (Stack Exchange; posts before 2018-05 CC BY-SA 3.0/2.5)"
        for qid, (title, body, date, tags, cands, prim, ans) in qs.items():
            top = sorted(ans, key=lambda x: -x[0])[:2]
            text = "\n\n".join([title, clean_html(body, keep_code)] + [clean_html(b, keep_code) for _, b in top])
            if len(text) < MIN:
                continue
            o.put(group=f"se/{site}/{qid}", register="qa", text=text[:MAX], source=f"stackexchange:{site}/{qid}",
                  source_revision=f"StackExchange dump 2024-04 (archive.org/details/stackexchange {z.name}), post {date}",
                  license=lic, redistribute=SA, created=date, cands=sorted(cands), primary=prim, tags=tags,
                  site=site)
        print(f"  {site}: {sum(o.n.values()) - n0} threads", flush=True)
    o.close("se")


# ---------------------------------------------------------------- arXiv CC-licensed full text (Common Pile)
def iter_abstracts(path):
    """gfissore/arxiv-abstracts-2021 rows (arxiv-abstracts.jsonl.gz): id, title, abstract, categories (list or str)."""
    for f in sorted(glob.glob(os.path.join(path, "**/*.jsonl.gz"), recursive=True)):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            c = d.get("categories")
            c = (c[0] if isinstance(c, list) and len(c) == 1 else " ".join(c) if isinstance(c, list) else c or "").split()
            yield d["id"], d.get("title") or "", d.get("abstract") or "", c


def arxiv_cats(path):
    return {i: c for i, _, _, c in iter_abstracts(path)}


def do_arxiv(a):
    cats = arxiv_cats(a.cats)
    print(f"  arXiv id->categories: {len(cats)}", flush=True)
    o = Out(Path(a.out) / "arxiv.jsonl")
    miss = 0
    for f in sorted(glob.glob(os.path.join(a.src, "*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            if d.get("created", "") >= M.CUTOFF:
                continue
            c = cats.get(d["id"])
            if not c:
                miss += 1
                continue
            nodes = [M.arxiv_node(x) for x in c]
            if M.arxiv_node(c[0]) is None or not any(nodes):
                continue
            body = d["text"]
            # skip title/abstract/front matter: start at the first section heading after ~800 chars if any
            m = re.search(r"\n#+ [^\n]{3,80}\n", body[800:])
            t = span(body, 800 + m.start() if m else 0)
            if not t:
                continue
            o.put(group=f"arxiv/{d['id']}", register="academic", text=t, source=f"arxiv:{d['id']}",
                  source_revision=f"common-pile/arxiv_papers {os.path.basename(f)}; categories gfissore/arxiv-abstracts-2021@e4c5fbd4",
                  license=d["metadata"]["license"], redistribute="attribution" if "Share" not in d["metadata"]["license"] else SA,
                  created=d.get("created"), cands=[n for n in nodes if n], primary=M.arxiv_node(c[0]), arxiv_cats=c)
        print(f"  {os.path.basename(f)}: {sum(o.n.values())} (no categories: {miss})", flush=True)
    o.close("arxiv")


def do_abstracts(a):
    """arXiv abstracts >= 1,100 chars (CC0 metadata); a second academic register where CC full text is thin."""
    o = Out(Path(a.out) / "abstracts.jsonl")
    for i, ti, ab, c in iter_abstracts(a.src):
        prim = M.arxiv_node(c[0]) if c else None
        ab = re.sub(r"\s+", " ", ab).strip()
        if prim is None or len(ab) < MIN:
            continue
        title = re.sub(r"\s+", " ", ti).strip()
        o.put(group=f"arxiv/{i}", register="abstract", text=f"{title}\n\n{ab}"[:MAX],
              source=f"arxiv-abs:{i}", source_revision="gfissore/arxiv-abstracts-2021@e4c5fbd4",
              license="CC0 1.0 (arXiv metadata)", redistribute="yes", created="<2022",
              cands=[M.arxiv_node(x) for x in c], primary=prim, arxiv_cats=c)
    o.close("abstracts")


# ---------------------------------------------------------------- LibreTexts / Pressbooks / OER / DOAB / news
LIBRE_NAV = re.compile(r"^.{0,400}?(Last updated\s+Save as PDF\s+Page ID\s+\d+\s*)", re.S)


def do_libre(a):
    o = Out(Path(a.out) / "libretexts.jsonl")
    for f in sorted(glob.glob(os.path.join(a.src, "*.gz"))):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            u = d["id"]
            m = re.match(r"https://([a-z0-9]+)\.libretexts\.org/(Bookshelves|Courses)/([^/]+)(?:/([^/]+))?", u)
            if not m or m.group(1) in ("med", "espanol") or "/Human_Biology" in u or d.get("created", "") >= M.CUTOFF:
                continue
            sub, kind, shelf, book = m.groups()
            node = M.LIBRE_SHELF.get(f"{sub}/{shelf}") if kind == "Bookshelves" else None
            base = M.LIBRE_SUB.get(sub)
            text = re.sub(r"\\\(\\PageIndex\{[^}]*\}\\\)|\\PageIndex\{[^}]*\}", "", LIBRE_NAV.sub("", d["text"], count=1))
            t = span(text)
            if not t or not (node or base):
                continue
            book_url = d["metadata"].get("book_url") or f"{sub}/{shelf}/{book}"
            o.put(group=f"libretexts/{book_url}", register="textbook", text=t, source=f"libretexts:{u}",
                  source_revision=f"common-pile/libretexts {os.path.basename(f)}", license=d["metadata"]["license"],
                  redistribute=SA if "Share" in d["metadata"]["license"] else "attribution", created=d.get("created"),
                  cands=[node, base], primary=node, title=d["metadata"].get("title"))
    o.close("libretexts")


def do_generic(a):
    """Pressbooks, OER Commons, DOAB, Common Pile news, peS2o: no subject label except peS2o fields."""
    name = a.name
    o = Out(Path(a.out) / f"{name}.jsonl")
    reg = {"pressbooks": "textbook", "oercommons": "textbook", "doab": "book", "news": "news", "pes2o": "academic"}[name]
    for f in sorted(glob.glob(os.path.join(a.src, "**/*.gz"), recursive=True)):
        for line in gzip.open(f, "rt"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            md = d.get("metadata") or {}
            created = str(d.get("created") or md.get("year") or "")
            if created and created >= M.CUTOFF:
                continue
            cands = ["*"]
            if name == "pes2o":
                fos = set(md.get("s2fieldsofstudy") or []) | set(md.get("extfieldsofstudy") or [])
                if fos & M.PES2O_DROP or not fos:
                    continue
                cands = [c for f_ in fos for c in M.PES2O_FIELD.get(f_, [])]
                if not cands:
                    continue
            lic = md.get("license") or md.get("oa_license") or ""
            body = d["text"]
            spans = [span(body, 600 if reg in ("academic", "book") else 0)]
            if reg == "book":
                step = max(MAX, len(body) // (a.book_chunks + 1))
                spans = [span(body, s) for s in range(step, len(body) - MIN, step)][: a.book_chunks]
            # one group per work: a book (DOAB book_id, Pressbooks book_url) or one article (news url); sections of a
            # book never span splits, unrelated news articles never share a group
            gid = (md.get("book_id") or md.get("book_url") or (md.get("url") if name == "news" else None)
                   or (str(d.get("id")).split(".")[0] if name == "doab" else None) or d.get("id") or md.get("url"))
            for k, t in enumerate(x for x in spans if x):
                o.put(group=f"{name}/{gid}", register=reg, text=t, source=f"{name}:{md.get('url') or gid}#{k}",
                      source_revision=f"common-pile/{name} {os.path.basename(f)}", license=lic,
                      redistribute=SA if "Share" in lic or "SA" in lic else "attribution", created=created,
                      cands=cands, primary=None, title=md.get("title"))
    o.close(name)


# ---------------------------------------------------------------- Project Gutenberg (Common Pile) + PG catalog LoCC
LOCC = [("BL", "hum.religion"), ("BM", "hum.religion"), ("BP", "hum.religion"), ("BQ", "hum.religion"),
        ("BR", "hum.religion"), ("BS", "hum.religion"), ("BT", "hum.religion"), ("BV", "hum.religion"),
        ("BX", "hum.religion"), ("D", "hum.history"), ("E", "hum.history"), ("F", "hum.history"),
        ("G1", "life.travel"), ("G", "soc.geography"), ("GV", "life.sports"), ("GN", "soc.anthropology"),
        ("GT", "soc.anthropology"), ("HB", "soc.economics"), ("HC", "soc.economics"), ("HD", "soc.business"),
        ("HF", "soc.business"), ("HG", "soc.economics.finance"), ("HJ", "soc.economics"), ("L", "soc.education"),
        ("TA", "eng.civil"), ("TJ", "eng.mechanical"), ("TK", "eng.electrical"), ("TP", "eng.chemical"),
        ("TH", "eng.civil"), ("TS", "eng.industrial"), ("TX", "life.cooking"), ("TT", "life.home_diy"),
        ("QA", "math"), ("QC", "sci.physics"), ("PN1993", "hum.film"), ("PN1995", "hum.film"), ("U", "hum.history.military"),
        ("V", "hum.history.military"), ("P9", "soc.communication"), ("PN4", "soc.communication")]


def locc_node(codes):
    out = []
    for c in codes:
        c = c.strip()
        best = None
        for pre, node in LOCC:
            if c.startswith(pre) and (best is None or len(pre) > len(best[0])):
                best = (pre, node)
        if best:
            out.append(best[1])
    return out


def do_gutenberg(a):
    cat = {}
    for r in csv.DictReader(open(a.catalog, encoding="utf-8")):
        if r.get("Language") == "en":
            cat[r["Text#"]] = (r.get("LoCC", "").split(";"), r.get("Subjects", ""), r.get("Title", ""))
    o = Out(Path(a.out) / "gutenberg.jsonl")
    for f in sorted(glob.glob(os.path.join(a.src, "**/*.gz"), recursive=True)):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            gid = str(d.get("id"))
            m = re.search(r"(\d+)$", gid)
            meta = cat.get(m.group(1)) if m else None
            if not meta:
                continue
            nodes = locc_node(meta[0])
            if not nodes:
                continue
            body = re.sub(r"\[(?:Illustration|Footnote|Sidenote|Transcriber'?s? [Nn]ote)[^\]]*\]|\bFIG\.\s*\d+\.?|\bFig\.\s*\d+\.?", "",
                          d["text"])                  # scan/edition markers are not subject text
            step = max(MAX * 2, len(body) // (a.book_chunks + 1))
            for k, s in enumerate(range(step, len(body) - MIN, step)):
                if k >= a.book_chunks:
                    break
                t = span(body, s)
                if t:
                    o.put(group=f"gutenberg/{m.group(1)}", register="book", text=t, source=f"gutenberg:{m.group(1)}#{k}",
                          source_revision=f"common-pile/project_gutenberg {os.path.basename(f)}; pg_catalog.csv LoCC",
                          license="Public domain (US; Project Gutenberg)", redistribute="yes", created="pre-1930",
                          cands=nodes, primary=nodes[0], title=meta[2][:200], subjects=meta[1][:300])
    o.close("gutenberg")


# ---------------------------------------------------------------- OpenWebMath / proof-pile / Wikivoyage / Rosetta
def do_owm(a):
    import pyarrow.parquet as pq
    o = Out(Path(a.out) / "openwebmath.jsonl")
    for f in sorted(glob.glob(os.path.join(a.src, "**/*.parquet"), recursive=True)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=5000, columns=["url", "date", "text"]):
            d = b.to_pydict()
            for u, dt, t in zip(d["url"], d["date"], d["text"]):
                if (dt or "") >= M.CUTOFF:
                    continue
                s = span(t)
                if s:
                    o.put(group=f"owm/{u}", register="web", text=s, source=f"openwebmath:{u}",
                          source_revision=f"open-web-math/open-web-math@fde8ef8d {os.path.basename(f)}",
                          license="ODC-By 1.0 (OpenWebMath; Common Crawl ToU)", redistribute="attribution",
                          created=dt, cands=["math", "sci.physics"], primary=None)
    o.close("openwebmath")


def do_proofpile(a):
    o = Out(Path(a.out) / "proofpile.jsonl")
    for f in sorted(glob.glob(os.path.join(a.src, "**/*.jsonl*"), recursive=True)):
        op = gzip.open if f.endswith(".gz") else open
        for line in op(f, "rt"):
            d = json.loads(line)
            meta = d.get("meta") or {}
            setn = meta.get("set_name") or os.path.basename(f)
            if "proofwiki" in f or "proofwiki" in str(setn):
                lic, part = "CC BY-SA 3.0 (ProofWiki)", "proofwiki"
            elif "stacks" in f or "stacks" in str(setn):
                lic, part = "GFDL 1.2 (Stacks Project)", "stacks"
            else:
                continue
            t = span(d["text"])
            if t:
                o.put(group=f"{part}/{hashlib.sha1(d['text'][:300].encode()).hexdigest()[:12]}", register="proof", text=t,
                      source=f"proof-pile:{part}", source_revision=f"hoskinson-center/proof-pile@490b9802 {os.path.basename(f)}",
                      license=lic, redistribute=SA, created="<2022", cands=["math"], primary=None)
    o.close("proofpile")


def do_wikivoyage(a):
    o = Out(Path(a.out) / "wikivoyage.jsonl")
    ns = None
    title, text = None, None
    for ev, el in ET.iterparse(bz2.open(a.src), events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "title":
            title = el.text
        elif tag == "ns":
            ns = el.text
        elif tag == "text":
            text = el.text or ""
        elif tag == "page":
            if ns == "0" and text and not text.lower().startswith("#redirect"):
                t = re.sub(r"\{\{[^{}]*\}\}", "", text)
                t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
                t = re.sub(r"'''?|<[^>]+>|^\*+\s*|^=+\s*|\s*=+$", "", t, flags=re.M)
                s = span(t)
                if s:
                    o.put(group=f"wikivoyage/{title}", register="guide", text=s, source=f"wikivoyage:{title}",
                          source_revision=os.path.basename(a.src), license="CC BY-SA 4.0 (Wikivoyage)", redistribute=SA,
                          created="dump", cands=["life.travel"], primary="life.travel")
            el.clear()
    o.close("wikivoyage")


# ---------------------------------------------------------------- Wikipedia by category membership (API crawl)
def do_wikicat(a):
    """Titles from a category crawl (amoeba_category_wikicrawl.json: {node: [titles]}) joined to the local enwiki
    20231101 extract (wiki.jsonl from amoeba_subject_sources.py)."""
    want = json.load(open(a.titles))
    by_title = defaultdict(list)
    for node, titles in want.items():
        for t in titles:
            by_title[t].append(node)
    o = Out(Path(a.out) / "wikicat.jsonl")
    for line in open(a.src):
        d = json.loads(line)
        nodes = by_title.get(d["title"])
        if nodes and len(d["text"]) >= MIN:
            o.put(group=f"enwiki/{d['id']}", register="encyclopedic", text=d["text"][:MAX], source=f"wikipedia:{d['title']}",
                  source_revision=f"enwiki 20231101 (wikimedia/wikipedia @b04c8d1c, {d['shard']})",
                  license="CC BY-SA 3.0 / GFDL (Wikipedia)", redistribute=SA, created="20231101", cands=nodes,
                  primary=nodes[0] if len(nodes) == 1 else None)
    o.close("wikicat")


# ---------------------------------------------------------------- code (shallow clones of permissive repos)
EXT = {".py": "prog.python", ".pyi": "prog.python.typing", ".pyx": "prog.python.cython", ".c": "prog.c", ".h": "prog.c",
       ".cc": "prog.cpp", ".cpp": "prog.cpp", ".hpp": "prog.cpp", ".m": "prog.objc", ".swift": "prog.swift",
       ".rs": "prog.rust", ".go": "prog.go", ".java": "prog.java", ".kt": "prog.kotlin", ".scala": "prog.scala",
       ".cs": "prog.csharp", ".fs": "prog.fsharp", ".js": "prog.javascript", ".mjs": "prog.javascript", ".ts": "prog.typescript",
       ".rb": "prog.ruby", ".pl": "prog.perl", ".pm": "prog.perl", ".pod": "prog.perl.pod", ".php": "prog.php", ".lua": "prog.lua",
       ".R": "prog.r", ".r": "prog.r", ".jl": "prog.julia", ".f90": "prog.fortran", ".f": "prog.fortran", ".F90": "prog.fortran",
       ".hs": "prog.haskell", ".ml": "prog.ocaml", ".mli": "prog.ocaml", ".lisp": "prog.lisp.common_lisp",
       ".el": "prog.lisp.emacs_lisp", ".scm": "prog.lisp.scheme", ".rkt": "prog.lisp.scheme", ".clj": "prog.lisp.clojure",
       ".erl": "prog.erlang", ".ex": "prog.elixir", ".exs": "prog.elixir", ".dart": "prog.dart", ".zig": "prog.zig",
       ".nim": "prog.nim", ".tcl": "prog.tcl", ".lean": "prog.lean", ".v": "prog.coq", ".pro": "prog.prolog",
       ".S": "prog.assembly", ".s": "prog.assembly", ".asm": "prog.assembly.x86", ".metal": "prog.gpu.metal",
       ".cu": "prog.gpu.cuda", ".cuh": "prog.gpu.cuda", ".glsl": "prog.gpu.glsl", ".vert": "prog.gpu.glsl", ".frag": "prog.gpu.glsl",
       ".comp": "prog.gpu.glsl", ".hlsl": "prog.gpu.hlsl", ".wgsl": "prog.gpu.wgsl", ".sh": "prog.shell", ".bash": "prog.shell.bash",
       ".zsh": "prog.shell.zsh", ".zsh-theme": "prog.shell.zsh", ".fish": "prog.shell.fish", ".ps1": "prog.shell.powershell",
       ".psm1": "prog.shell.powershell", ".bat": "prog.shell.batch", ".cmd": "prog.shell.batch", ".sql": "prog.sql",
       ".html": "prog.web.html", ".css": "prog.web.css", ".scss": "prog.web.scss", ".jinja": "prog.web.templates",
       ".j2": "prog.web.templates", ".hbs": "prog.web.templates", ".handlebars": "prog.web.templates", ".md": "prog.markup.markdown",
       ".rst": "prog.markup.rst", ".tex": "prog.markup.latex", ".adoc": "prog.markup.asciidoc", ".1": "prog.markup.man",
       ".json": "prog.data.json", ".yaml": "prog.data.yaml", ".yml": "prog.data.yaml", ".toml": "prog.data.toml",
       ".xml": "prog.data.xml", ".csv": "prog.data.csv", ".ini": "prog.data.ini", ".proto": "prog.data.protobuf",
       ".cmake": "prog.build.cmake", ".bzl": "prog.build.bazel", ".nix": "prog.build.nix", ".tf": "prog.build.hcl",
       ".hcl": "prog.build.hcl"}
NAMES = {"Makefile": "prog.build.make", "GNUmakefile": "prog.build.make", "CMakeLists.txt": "prog.build.cmake",
         "Dockerfile": "prog.build.dockerfile", "BUILD": "prog.build.bazel", "BUILD.bazel": "prog.build.bazel",
         "WORKSPACE": "prog.build.bazel"}
SKIPDIR = re.compile(r"/(?:\.git|node_modules|vendor|third_party|thirdparty|dist|build|\.venv|site-packages|testdata|fixtures?)/")
SPDX = [("Apache License", "Apache-2.0"), ("MIT License", "MIT"), ("Permission is hereby granted, free of charge", "MIT"),
        ("BSD 3-Clause", "BSD-3-Clause"), ("Redistribution and use in source and binary forms", "BSD"),
        ("ISC License", "ISC"), ("Mozilla Public License", "MPL-2.0"), ("Eclipse Public License", "EPL"),
        ("GNU AFFERO GENERAL PUBLIC", "AGPL"), ("GNU LESSER GENERAL PUBLIC", "LGPL"), ("GNU GENERAL PUBLIC LICENSE", "GPL"), ("Artistic License", "Artistic-2.0"),
        ("The Unlicense", "Unlicense"), ("CeCILL-B", "CeCILL-B"), ("Boost Software License", "BSL-1.0"),
        ("This is free and unencumbered", "Unlicense"), ("curl license", "curl"), ("COPYRIGHT AND PERMISSION NOTICE", "curl"),
        ("Tcl/Tk License", "TCL"), ("This software is copyrighted by the Regents", "TCL")]


def repo_license(d):
    for n in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING", "epl-v10.html", "LICENSE.html", "license.txt", "COPYING.md", "license.terms", "LICENSE-MIT",
              "LICENSE.APACHE2", "LICENSE-APACHE"):
        p = d / n
        if p.exists():
            t = p.read_text(errors="replace")[:4000]
            for k, v in SPDX:
                if k.lower() in t.lower():
                    return v, n
            return f"see {n}", n
    return "unknown", None


def do_code(a):
    o = Out(Path(a.out) / "code.jsonl")
    lic_report = {}
    for d in sorted(Path(a.src).iterdir()):
        if not (d / "COMMIT").exists():      # probe corpora: any snapshot is fine (Pierre 2026-11-18: probing, not training)
            continue
        lic, lf = repo_license(d)
        head = (d / "COMMIT").read_text().strip()[:12]
        lic_report[d.name] = {"license": lic, "file": lf, "commit": head}
        if lic == "unknown" or lic.startswith("see "):
            print(f"  skip {d.name}: licence not recognised ({lic})", flush=True)
            continue
        per = Counter()
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name.startswith("._") or SKIPDIR.search("/" + str(p.relative_to(d)) + "/"):
                continue
            node = NAMES.get(p.name) or EXT.get(p.suffix)
            if not node or ".min." in p.name:
                continue
            if per[node] >= a.per_repo:
                continue
            try:
                t = p.read_text(errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            if len(t) < MIN or len(t) > 400_000 or max(map(len, t.splitlines() or [""])) > 1000:
                continue
            rel = str(p.relative_to(d))
            k = (h := int(hashlib.sha256(rel.encode()).hexdigest()[:6], 16)) % max(1, len(t) // MAX + 1)
            body = t[k * MAX: k * MAX + MAX] if len(t) > MAX else t
            cut = body.rfind("\n", 0, MAX)
            body = body[: cut if cut > MIN else MAX]
            if len(body) < MIN:
                continue
            per[node] += 1
            top = rel.split("/")[0] if "/" in rel else "."
            o.put(group=f"code/{d.name}/{top}", register="code", text=f"// {rel}\n{body}" if a.path_header else body,
                  source=f"github:{d.name.replace('_', '/', 1)}/{rel}", source_revision=f"github tarball @{head} ({'last commit before 2022-12-01' if (d / 'CUTOFF_COMMIT').exists() else 'HEAD'}; fetched 2026-11-18)",
                  license=lic, redistribute={"GPL": "copyleft", "AGPL": "copyleft", "LGPL": "copyleft", "CeCILL-B": "attribution",
                                                     "MPL-2.0": "copyleft (file-level)", "EPL": "copyleft (weak)"}.get(lic, "attribution"),
                  created="", cands=[node], primary=node, ext=p.suffix or p.name)
        print(f"  {d.name} [{lic}]: {sum(per.values())} files {dict(per.most_common(4))}", flush=True)
    json.dump(lic_report, open(Path(a.out) / "code_licenses.json", "w"), indent=1)
    o.close("code")


# ---------------------------------------------------------------- Rosetta Code (GFDL 1.2), extra code register
ROSETTA = {"python": "prog.python", "c": "prog.c", "c++": "prog.cpp", "objective-c": "prog.objc", "swift": "prog.swift",
           "rust": "prog.rust", "go": "prog.go", "java": "prog.java", "kotlin": "prog.kotlin", "scala": "prog.scala",
           "c sharp": "prog.csharp", "c#": "prog.csharp", "f sharp": "prog.fsharp", "f#": "prog.fsharp",
           "javascript": "prog.javascript", "typescript": "prog.typescript", "ruby": "prog.ruby", "perl": "prog.perl",
           "php": "prog.php", "lua": "prog.lua", "r": "prog.r", "julia": "prog.julia", "matlab": "prog.matlab",
           "octave": "prog.matlab", "fortran": "prog.fortran", "haskell": "prog.haskell", "ocaml": "prog.ocaml",
           "common lisp": "prog.lisp.common_lisp", "emacs lisp": "prog.lisp.emacs_lisp", "scheme": "prog.lisp.scheme",
           "racket": "prog.lisp.scheme", "clojure": "prog.lisp.clojure", "erlang": "prog.erlang", "elixir": "prog.elixir",
           "dart": "prog.dart", "zig": "prog.zig", "nim": "prog.nim", "tcl": "prog.tcl", "lean": "prog.lean",
           "coq": "prog.coq", "prolog": "prog.prolog", "x86 assembly": "prog.assembly.x86",
           "arm assembly": "prog.assembly.arm64", "unix shell": "prog.shell.sh", "bash": "prog.shell.bash",
           "zsh": "prog.shell.zsh", "powershell": "prog.shell.powershell", "batch file": "prog.shell.batch",
           "sql": "prog.sql", "pl/sql": "prog.sql", "awk": "prog.shell"}


def do_rosetta(a):
    import pyarrow.parquet as pq
    o = Out(Path(a.out) / "rosetta.jsonl")
    for f in sorted(glob.glob(os.path.join(a.src, "**/*.parquet"), recursive=True)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=5000):
            for r in b.to_pylist():
                node = ROSETTA.get(str(r.get("language_name", "")).strip().lower())
                code = r.get("code") or ""
                if not node or len(code) < MIN:
                    continue
                o.put(group=f"rosetta/{r.get('task_name')}/{r.get('language_name')}", register="rosetta",
                      text=code[:MAX], source=f"rosettacode:{r.get('task_name')}/{r.get('language_name')}",
                      source_revision=f"christopher/rosetta-code@11d8b38c {os.path.basename(f)}",
                      license="GFDL 1.2 (Rosetta Code)", redistribute=SA, created="", cands=[node], primary=node)
    o.close("rosetta")


# ---------------------------------------------------------------- cybersecurity corpus (amoeba-corpus cyber build)
# Register = source family (map: --register-map). Rules from that build (surveys/cyber-build-20261118.md): skip dup_of rows, *.ndjson metadata,
# news.general and synthetic rows; eval_guard rows are written to a separate guard file (dedup only, never boxed);
# groups are source + document id. Expert sources keep their leaf label; keyword-labelled sets (filtered web,
# arXiv) only nominate `sec` (or their intel root) and the log-odds descent picks the leaf.
# The source -> register map lives in the private corpus repo (sources/category-sources-20261118/
# cyber-register-map.json): some sources there are private_corpus and are never named in mlx2.


def do_cyber(a):
    cmap = json.load(open(a.register_map))
    CYBER_REGISTER = [(x["match"], x["register"]) for x in cmap["registers"]]
    CYBER_ROUGH = re.compile(cmap["rough_label_sources"])
    o = Out(Path(a.out) / "cyber.jsonl")
    guard = open(Path(a.out) / "cyber-eval-guard.shingles", "w")
    skipped = Counter()
    rng_keep = {}
    for d in sorted(Path(a.src).glob("*-20261118")):
        if not (d / "SOURCE.json").exists():
            continue
        name = d.name
        reg = next((r for rx, r in CYBER_REGISTER if re.match(rx, name)), None)
        guard_src = name.startswith("cyber-evals-guard")
        if not reg and not guard_src:
            skipped[f"no register: {name}"] += 1
            continue
        for f in sorted(d.glob("*.jsonl")):
            if f.name.startswith("EXCLUDED"):
                continue
            for line in open(f):
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                t = r.get("text") or ""
                if guard_src or r.get("node") == "eval_guard":
                    w = re.findall(r"\w+", t.lower())          # 13-word shingles, hashed (builder drops overlaps)
                    for k in range(0, max(1, len(w) - 12)):
                        guard.write(hashlib.blake2b(" ".join(w[k:k + 13]).encode(), digest_size=8).hexdigest() + "\n")
                    continue
                node = r.get("node") or ""
                if r.get("dup_of") or node.startswith("news") or r.get("origin") == "synthetic" or \
                        (r.get("lang") and r["lang"] != "en"):
                    skipped["dup/news/synthetic/non-en"] += 1
                    continue
                if name.startswith("stackexchange-sec") and r.get("site") in ("serverfault", "unix", "networkengineering"):
                    skipped["SE site covered by the full-site extraction"] += 1
                    continue                      # one copy per thread: these sites come from se-<site>.jsonl
                if not node.startswith(("sec", "intel")):
                    skipped["node outside sec/intel"] += 1
                    continue
                gid = f"{name}/{r.get('id')}"
                if reg == "web" and int(hashlib.sha256(("keep\0" + gid).encode()).hexdigest()[:8], 16) % a.web_keep_mod:
                    continue                      # subsample the 810k filtered-web rows (salted: independent of split_of)
                s_ = span(t)
                if not s_:
                    continue
                rough = bool(CYBER_ROUGH.match(name))
                root = node.split(".")[0]
                o.put(group=gid, register=reg, text=s_, source=f"{name}:{r.get('id')}",
                      source_revision=r.get("source_revision") or name, license=r.get("license") or "",
                      redistribute=r.get("redistribute") or "", created=str(r.get("date") or ""),
                      cands=[root] if rough else [node], primary=None if rough else node,
                      private_corpus=str(r.get("private_corpus")) == "True", node_label=node, url=r.get("url"))
        print(f"  {name} [{reg or 'guard'}]: {sum(o.n.values())} so far", flush=True)
    guard.close()
    print(f"cyber skipped: {dict(skipped)}", flush=True)
    o.close("cyber")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["se", "arxiv", "abstracts", "libre", "generic", "gutenberg", "owm", "proofpile",
                                     "wikivoyage", "wikicat", "code", "rosetta", "cyber"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cats", default="", help="arxiv: gfissore/arxiv-abstracts-2021 dir")
    ap.add_argument("--catalog", default="", help="gutenberg: pg_catalog.csv")
    ap.add_argument("--titles", default="", help="wikicat: crawl json")
    ap.add_argument("--name", default="", help="generic: pressbooks|oercommons|doab|news|pes2o")
    ap.add_argument("--only", default="", help="se: comma list of sites")
    ap.add_argument("--trust", default="", help="se: sites copied from the 2026-11-16 download (no .done marker)")
    ap.add_argument("--per-cat", type=int, default=15000)
    ap.add_argument("--so-per-cat", type=int, default=8000)
    ap.add_argument("--book-chunks", type=int, default=12)
    ap.add_argument("--per-repo", type=int, default=1500, help="code: files per (repo, node)")
    ap.add_argument("--path-header", action="store_true", help="code: prefix the file path")
    ap.add_argument("--register-map", default="", help="cyber: register map json (private corpus repo)")
    ap.add_argument("--web-keep-mod", type=int, default=8, help="cyber: keep 1 in N filtered-web rows")
    ap.add_argument("--created-before", default="", help="optional date cut (YYYY-MM-DD); default none")
    a = ap.parse_args()
    if a.created_before:
        M.CUTOFF = a.created_before
    a.trust = set(a.trust.split(",")) if a.trust else set()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"se": do_se, "arxiv": do_arxiv, "abstracts": do_abstracts, "libre": do_libre, "generic": do_generic,
     "gutenberg": do_gutenberg, "owm": do_owm, "proofpile": do_proofpile, "wikivoyage": do_wikivoyage,
     "wikicat": do_wikicat, "code": do_code, "rosetta": do_rosetta, "cyber": do_cyber}[a.what](a)


if __name__ == "__main__":
    main()
