"""Data for the Hy4-flash zh/ja/hi box (docs/research/amoeba-hy4-cjk-hin-20261121/PLAN.md), from local corpora only.

CPU only, no downloads (Pierre 2026-11-21: "We have tons of language corpus, use it"). Reads the amoeba-corpus builds
and the vendored GSM8K, writes into --out (refuses a nonempty one):

  zh.jsonl ja.jsonl hi.jsonl   extract --extra-topic-corpus rows (lang.cmn / lang.jpn / lang.hin): the lang-prose
                               fit split -> split fit (the box set takes its val from these by group hash), held ->
                               held. Its val and held2 splits are kept out for the removal check.
  core-code.jsonl core-math.jsonl
                               extra core corpora (domain code / math, split fit): category-boxes prog.* / math.* fit
                               rows and GSM8K train (all but every 7th row by a seeded hash)
  removal-nontarget-val.jsonl removal-nontarget-held.jsonl
                               classes eng / code / math. val: lang-prose-eng val, category-boxes prog.*/math.* val,
                               GSM8K train every-7th; held: topic-probes-9a1f35475 lang.eng held (Wikipedia, PG-19),
                               lang-prose-eng held, category-boxes prog.*/math.* held, GSM8K test
  removal-target-val.jsonl removal-target-held.jsonl
                               lang-prose val / held2 of the three languages (class lang.cmn / lang.jpn / lang.hin)
  private-corpus-ids.json      extract --exclude-texts: probe-corpus rows of a restrictive-licence source (licence
                               marked internal research use / redistribute no) in the three languages
  inventory.json               rows and (with --tokenizer) tokens per file, class, split and licence

Every row keeps ``origin``, ``license``, ``source_revision`` and ``group``. Rows whose source may not be
redistributed are labelled ``private_corpus`` (their source name is replaced by a digest; the build manifests in the
amoeba-corpus repository keep the mapping): they are used for carving and checks only, never in a shipped prompt set.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path.home() / "Desktop/amoeba-corpus/builds"
LANG = {"cmn": ("lang.cmn", "lang-prose-cmn-48c93e8a0", "zh"), "jpn": ("lang.jpn", "lang-prose-jpn-c54a12789", "ja"),
        "hin": ("lang.hin", "lang-prose-hin-c54a12789", "hi")}
ENG = "lang-prose-eng-c7af39fdf"
BOXES = "category-boxes-8ec12cb93"
PROBES = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/topic-probes-e8c22f31a/corpus.jsonl"
PROBES_ENG = PROBES.parents[1] / "topic-probes-9a1f35475/corpus.jsonl"     # the larger English held split
GSM8K = Path(__file__).resolve().parents[1] / "benchmarks/gsm8k"


def jl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        for x in f:
            if x.strip():
                yield json.loads(x)


def h(s, seed=0) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{s}".encode()).digest()[:8], "big")


def private(r) -> bool:
    red = str(r.get("redistribute", "")).lower()
    lic = str(r.get("license", "")).lower()
    return red == "no" or "internal research use" in lic


PRIVATE_LICENSE = "private_corpus (not redistributable; internal carving and checks only)"


def _pc(v) -> str:
    return "private_corpus:" + hashlib.sha256(str(v).encode()).hexdigest()[:16]


def row(r, text=None, **kw):
    """One output row. A private_corpus row keeps only its text, origin and split fields: its licence becomes the
    neutral PRIVATE_LICENSE and every identifying field (source, group, doc id, row id, source revision) a digest, so
    no private source name reaches any file (the amoeba-corpus manifests keep the mapping)."""
    out = {"text": text if text is not None else r["text"], "origin": r.get("origin", "natural"),
           "license": r.get("license"), "source_revision": r.get("source_revision"),
           "group": r.get("group") or r.get("doc_id") or r.get("id"), "doc_id": r.get("doc_id") or r.get("id")}
    if private(r):
        out.update(private_corpus=True, redistribute="no", license=PRIVATE_LICENSE, source=_pc(r.get("source", "")),
                   source_revision=None, group=_pc(out["group"]), doc_id=_pc(out["doc_id"]))
        if "id" in kw:
            kw = {**kw, "id": _pc(kw["id"])}
    else:
        out.update(private_corpus=False, source=r.get("source"), redistribute=r.get("redistribute"))
    return {**out, **kw}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-core", type=int, default=1500, help="code core-corpus rows")
    ap.add_argument("--math-core", type=int, default=1500, help="math core-corpus rows (plus GSM8K train)")
    ap.add_argument("--per-class-val", type=int, default=300)
    ap.add_argument("--per-class-held", type=int, default=300)
    ap.add_argument("--tokenizer", default="", help="artifact dir: count tokens with its tokenizer")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    files = defaultdict(list)
    # target languages
    for lang, (topic, build, short) in LANG.items():
        for r in jl(CORPUS / build / "corpus.jsonl"):
            sp = r["split"]
            if sp in ("fit", "held"):
                files[f"{short}.jsonl"].append(row(r, id=f"lp-{lang}-{r['id']}", topics=[topic], split=sp))
            elif sp == "val":
                files["removal-target-val.jsonl"].append(row(r, domain=topic))
            elif sp == "held2":
                files["removal-target-held.jsonl"].append(row(r, domain=topic))
    # English
    for r in jl(CORPUS / ENG / "corpus.jsonl"):
        if r["split"] in ("val", "held"):
            files[f"removal-nontarget-{r['split']}.jsonl"].append(row(r, domain="eng"))
    eng = [r for r in jl(PROBES_ENG) if "lang.eng" in r["topics"] and r["split"] == "held"
           and not any(t.startswith(("prog", "math", "lang.cmn", "lang.jpn", "lang.hin")) for t in r["topics"])
           and not private(r)]
    eng.sort(key=lambda r: h(r["id"], a.seed))
    files["removal-nontarget-held.jsonl"] += [row(r, domain="eng") for r in eng[:a.per_class_held]]
    # code and math from the category boxes (fit -> core, val -> removal val, held -> removal held)
    pools = defaultdict(list)
    for r in jl(CORPUS / BOXES / "corpus.jsonl.gz"):
        fam = r["node"].split(".")[0]
        if fam in ("prog", "math") and r["split"] in ("fit", "val", "held"):
            pools[(("code" if fam == "prog" else "math"), r["split"])].append(r)
    for (cls, sp), rs in pools.items():
        rs.sort(key=lambda r: h(r["id"], a.seed))
        if sp == "fit":
            n = a.code_core if cls == "code" else a.math_core
            files[f"core-{cls}.jsonl"] += [row(r, domain=cls, split="fit") for r in rs[:n]]
        else:
            n = a.per_class_val if sp == "val" else a.per_class_held
            files[f"removal-nontarget-{sp}.jsonl"] += [row(r, domain=cls) for r in rs[:n]]
    # GSM8K (MIT, vendored)
    lic = {"license": "MIT (GSM8K)", "origin": "natural", "source_revision": "openai/gsm8k main", "redistribute": "yes"}
    for i, r in enumerate(jl(GSM8K / "train.jsonl")):
        t = f"Question: {r['question']}\nAnswer: {r['answer']}"
        base = {**lic, "source": f"gsm8k:train:{i}", "id": f"gsm8k-train-{i}", "group": f"gsm8k/train/{i}"}
        if h(base["id"], a.seed) % 7 == 0:
            files["removal-nontarget-val.jsonl"].append(row(base, text=t, domain="math"))
        else:
            files["core-math.jsonl"].append(row(base, text=t, domain="math", split="fit"))
    for i, r in enumerate(jl(GSM8K / "test.jsonl")):
        base = {**lic, "source": f"gsm8k:test:{i}", "id": f"gsm8k-test-{i}", "group": f"gsm8k/test/{i}"}
        files["removal-nontarget-held.jsonl"].append(row(base, text=f"Question: {r['question']}\nAnswer: {r['answer']}",
                                                         domain="math"))
    # restrictive-licence probe rows of the three languages: excluded from every stage
    excl = [r["id"] for r in jl(PROBES) if r["topics"] and r["topics"][0] in ("lang.cmn", "lang.jpn", "lang.hin",
                                                                              "lang.zho") and private(r)]
    (out / "private-corpus-ids.json").write_text(json.dumps({"probe_ids": excl}, indent=1))
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    inv = {}
    for name, rs in sorted(files.items()):
        with open(out / name, "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter((r.get("domain") or r.get("topics", ["?"])[0], r.get("split", "-"), r["private_corpus"]) for r in rs)
        rec = {"rows": len(rs), "by_class_split_private": {f"{k[0]}|{k[1]}|{'private' if k[2] else 'open'}": v
                                                           for k, v in sorted(c.items())}}
        if tok is not None:
            t = Counter()
            for r in rs:
                t[r.get("domain") or r["topics"][0]] += len(tok.encode(r["text"], add_special_tokens=False))
            rec["tokens"] = dict(t)
        inv[name] = rec
    inv["private-corpus-ids.json"] = {"probe_ids": len(excl)}
    (out / "inventory.json").write_text(json.dumps(inv, indent=1))
    print(json.dumps(inv, indent=1))


if __name__ == "__main__":
    main()
