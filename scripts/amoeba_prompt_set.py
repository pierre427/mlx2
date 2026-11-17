#!/usr/bin/env python3
"""Build a tile set's PROMPT SET (Pierre 2026-11-16, docs/research/2026-11-13-amoeba-guide/mapping-workflow.md §10): one
Parquet file holding every text used in mapping and carving, plus a JSON manifest (schema, sha256, counts). CPU only.

Inputs (repeatable):
  --carve-run DIR   a research.amoeba.carve run dir with inputs.jsonl (every sequence the run used: split, domain,
                    loss mask, and its source rows: file corpus/held/val, row key, doc id, token offset, window length)
                    and carve.json (corpus / held / val file paths); --stage NAME labels the run (e.g. box:lang_cmn)
  --atlas-run DIR   a carve atlas run (carve.json + atlas/meta.json seq_docs): every captured sequence's document
  --probe-corpus F  the probe corpus jsonl (fills origin / licence / source for rows that do not carry them, by probe id)
One row per (stage, sequence, source claim): stage, split (mean / fit / spare / val / held; atlas), domain, topic,
topics, doc_id, probe_id, token_offset, window_tokens, loss_from, text (the source row's full text), for continuations:
prompt, continuation, mode, temperature, seed; origin, source, source_revision (corpus build), license, method.
Needs pyarrow (e.g. the mlx-uag venv's python); everything else is the standard library.

Usage: amoeba_prompt_set.py --out DIR --name NAME [--carve-run DIR --stage S]... [--atlas-run DIR --stage S]...
                            [--probe-corpus F] [--checkpoint C] [--taxonomy-version V]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split("\n") if x.strip()]


def row_key(r):
    if r.get("prompt") is not None:
        return f"gen:{r.get('prompt_key')}:{r.get('mode')}:{r.get('seed')}"
    return r.get("probe_id") or r.get("doc_id") or hashlib.sha256(r["text"].encode()).hexdigest()[:16]


def doc_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--carve-run", action="append", default=[])
    ap.add_argument("--atlas-run", action="append", default=[])
    ap.add_argument("--capture-run", action="append", default=[],
                    help="a map topic capture (items.json + meta.json: every probe text's first seq_len tokens); needs "
                         "--probe-corpus for the texts")
    ap.add_argument("--stage", action="append", default=[], help="one per run, in the order given (carve runs, then atlas runs, then capture runs)")
    ap.add_argument("--probe-corpus", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--taxonomy-version", default="")
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    probes = {}
    build = ""
    if a.probe_corpus:
        build = Path(a.probe_corpus).parent.name
        for r in read_jsonl(a.probe_corpus):
            probes[r["id"]] = r
    stages = list(a.stage)
    runs = ([("carve", d) for d in a.carve_run] + [("atlas", d) for d in a.atlas_run] +
            [("capture", d) for d in a.capture_run])
    if stages and len(stages) != len(runs):
        raise SystemExit("--stage must be given once per run")
    cols = {k: [] for k in ("stage", "split", "use", "trained", "text_cut_chars", "domain", "topic", "topics", "doc_id", "probe_id", "token_offset",
                            "window_tokens", "loss_from", "text", "prompt", "continuation", "mode", "temperature",
                            "seed", "origin", "source", "source_revision", "license", "method")}

    def prov(r):
        p = probes.get(r.get("probe_id") or "", {})
        return {"origin": r.get("origin") or p.get("origin"), "license": r.get("license") or p.get("license"),
                "source": r.get("probe_source") or p.get("source") or r.get("source"),
                "source_revision": r.get("corpus_build") or (build if p else None), "method": r.get("method") or p.get("method")}

    def emit(stage, split, dn, r, off, length, lf, use=None, trained=None, cut=0):
        pv = prov(r)
        if r.get("prompt") is not None:                  # a continuation: provenance of its source document
            src = sources_by_doc.get(r.get("doc_id"), {})
            spv = prov(src) if src else {}
            pv = {**spv, "origin": f"model_continuation({spv.get('origin')})", "method": "onpolicy_gen"}
        topics = r.get("topics") or (probes.get(r.get("probe_id") or "", {}).get("topics"))
        vals = {"stage": stage, "split": split, "use": use, "trained": trained, "text_cut_chars": cut, "domain": dn, "topic": (topics or [None])[0],
                "topics": json.dumps(topics) if topics else None, "doc_id": r.get("doc_id") or doc_id(r["text"]),
                "probe_id": r.get("probe_id"), "token_offset": off, "window_tokens": length, "loss_from": lf,
                "text": r["text"], "prompt": r.get("prompt"),
                "continuation": r["text"][len(r["prompt"]):] if r.get("prompt") is not None else None,
                "mode": r.get("mode"), "temperature": r.get("temperature"), "seed": r.get("seed"), **pv}
        for k in cols:
            cols[k].append(vals.get(k))
    sources_by_doc = {}
    for i, (kind, d) in enumerate(runs):
        d = Path(d)
        stage = stages[i] if stages else f"{kind}:{d.name}"
        if kind == "capture":                            # map capture: probe ids, first seq_len tokens of each text
            meta = json.loads((d / "meta.json").read_text())
            L = int((meta.get("args") or {}).get("seq_len", 64))
            for it in json.loads((d / "items.json").read_text()):
                r = probes.get(it["id"])
                if r is None:
                    raise SystemExit(f"{d}: probe {it['id']} not in --probe-corpus")
                emit(stage, it["split"], "probe", {**r, "probe_id": r["id"]}, 0, int(it.get("n_tokens") or L), 0,
                     "map", False)
            continue
        cj = json.loads((d / "carve.json").read_text())
        args = cj.get("args", {})
        files = {"corpus": args.get("corpus", "").removeprefix("jsonl:"), "held": args.get("held_file") or "",
                 "val": args.get("val_file") or ""}
        rows = {}
        for f, p in files.items():
            if p and Path(p).exists():
                for r in read_jsonl(p):
                    rows[(f, row_key(r))] = r
                    if r.get("prompt") is None:
                        sources_by_doc.setdefault(r.get("doc_id") or doc_id(r["text"]), r)
        if kind == "carve":
            if not (d / "inputs.jsonl").exists():
                raise SystemExit(f"{d}: no inputs.jsonl (run predates the prompt-set record)")
            for x in read_jsonl(d / "inputs.jsonl"):
                for c in x["sources"]:
                    r = rows.get((c["file"], c["row"]))
                    if r is None:
                        raise SystemExit(f"{d}: source row {c['file']}:{c['row']} not found")
                    emit(stage, x["split"], x["domain"], r, c["off"], c["len"], x["loss_from"], x.get("use"),
                         x.get("trained"), c.get("text_cut_chars", 0))
        else:
            meta = json.loads((Path(args.get("atlas_dir") or d / "atlas") / "meta.json").read_text())
            by = {}
            for (f, _), r in rows.items():
                if f == "corpus":
                    by.setdefault(r.get("doc_id") or doc_id(r["text"]), r)
            L = int(args.get("seq_len", 64))
            for claims in meta.get("seq_docs") or []:
                for doc, off in claims:
                    r = by.get(doc)
                    if r is not None:
                        emit(stage, "atlas", r.get("domain"), r, off, L, 0, "atlas")
    t = pa.table(cols)
    pf = out / f"{a.name}.parquet"
    pq.write_table(t, pf, compression="zstd")
    sha = hashlib.sha256(pf.read_bytes()).hexdigest()
    man = {"schema": "mlx2.amoeba-prompt-set.v1", "name": a.name, "file": pf.name, "sha256": sha, "rows": t.num_rows,
           "columns": {f.name: str(f.type) for f in t.schema}, "checkpoint": a.checkpoint or None,
           "taxonomy_version": a.taxonomy_version or None, "probe_corpus": a.probe_corpus or None,
           "runs": [{"kind": k, "dir": str(d), "stage": (stages[i] if stages else None)} for i, (k, d) in enumerate(runs)],
           "counts": {"by_stage": Counter(cols["stage"]), "by_split": Counter(cols["split"]), "by_use": Counter(cols["use"]),
                      "by_origin": Counter(str(x) for x in cols["origin"]),
                      "by_license": Counter(str(x) for x in cols["license"]),
                      "documents": len(set(cols["doc_id"]))}}
    (out / f"{a.name}.manifest.json").write_text(json.dumps(man, indent=1, default=dict))
    print(json.dumps({k: man[k] for k in ("rows", "sha256")}))


if __name__ == "__main__":
    sys.exit(main())
