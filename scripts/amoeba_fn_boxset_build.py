#!/usr/bin/env python3
"""FN box set from the taxonomy probe corpus (Pierre 2026-11-16, step 3: boxes, the more the better), CPU only.

Boxes: one per language (lang.* primary topic; the manifest's macrolanguage aliases already folded) and one per
depth-2 subject branch (e.g. sci.physics = every probe whose primary topic is sci.physics or below) with at least
--min-fit fit probes. --boxes restricts to a comma-separated list (batch order is the caller's).

Per box, from the probes' own fit/held split (group hash, so a group never spans splits):
  docs.jsonl   carve documents: fit probes, minus the val groups (domain "box");
  val.jsonl    export-point validation documents: fit probes of 1 in --val-every groups (domain "box");
  held.jsonl   the held test set: every held probe (domain "box"), never carved on, never used for selection;
  prompts_{carve,val,held}.jsonl  the documents whose mid-document continuations the GPU stage generates (seeded
               choice, --carve-prompts / --val-prompts / --held-prompts per box).
Writes OUT/boxes.json (box, kind, topics, counts) and OUT/<box>/... Never model-generated held text: continuations are
added as their own gen:box:<mode> domains, next to the probes, and are reported as such.

Usage: amoeba_fn_boxset_build.py --corpus CORPUS.jsonl --out DIR [--boxes a,b] [--min-fit 32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def h(s, seed=0):
    return int(hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()[:12], 16)


def box_of(topic):
    if topic.startswith("lang."):
        return topic
    return ".".join(topic.split(".")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", default="")
    ap.add_argument("--topic", default="", help="one box for this taxonomy id: every probe whose primary topic is it "
                    "or below it (targeted extraction, mapping-workflow s11); replaces the branch/language grouping")
    ap.add_argument("--min-fit", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--carve-prompts", type=int, default=16)
    ap.add_argument("--val-prompts", type=int, default=8)
    ap.add_argument("--held-prompts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="", help="tokenizer for the minimum-count gate (windows that reach --seq-len)")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--prompt-min", type=int, default=112, help="tokens a document needs to seed a continuation")
    ap.add_argument("--min-carve", type=int, default=52, help="carve windows (2 per document of >= --seq-len)")
    ap.add_argument("--min-val", type=int, default=16, help="val windows + 4 x val continuations")
    ap.add_argument("--val-windows", type=int, default=2, help="windows per val document (the carve's --val-windows; "
                    "coordinator 2026-11-16: 4 for thin val splits)")
    ap.add_argument("--min-held", type=int, default=16, help="held windows + 4 x held continuations")
    ap.add_argument("--manifest", default="", help="probe-corpus-manifest.json: summary.held_synthetic_topics / "
                    "held_lab_topics flag boxes whose held split is mostly synthetic or only from our own repos")
    a = ap.parse_args()
    from research.amoeba.carve.corpus import _doc_id, sha12
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(x) for x in Path(a.corpus).read_text().split("\n") if x.strip()]
    build = Path(a.corpus).parent.name                         # e.g. topic-probes-e8c22f31a
    by = defaultdict(list)
    for r in rows:
        t0 = r["topics"][0]
        if a.topic:                      # one id, or several joined by '+' (one box from all of them)
            if any(t0 == n or t0.startswith(n + ".") for n in a.topic.split("+") if n):
                by[a.topic].append(r)
            continue
        by[box_of(t0)].append(r)
    want = [a.topic] if a.topic else ([b for b in a.boxes.split(",") if b] or sorted(by))
    flags = {"held_synthetic": set(), "held_lab": set()}
    if a.manifest:
        summ = json.loads(Path(a.manifest).read_text()).get("summary", {})
        flags = {"held_synthetic": set(summ.get("held_synthetic_topics") or []),
                 "held_lab": set(summ.get("held_lab_topics") or [])}
    boxes, extend = [], []
    tok = None
    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    ntok = lambda t: len(tok.encode(t, add_special_tokens=False)) if tok else 10 ** 6    # noqa: E731
    for b in want:
        rs = by.get(b, [])
        fit = [r for r in rs if r["split"] == "fit"]
        if len(fit) < a.min_fit:
            print(f"skip {b}: {len(fit)} fit probes < {a.min_fit} (probes to extend)")
            extend.append({"node": b, "fit_probes": len(fit), "short": ["fit_probes"]})
            continue
        name = re.sub(r"[^a-z0-9]+", "_", b.lower()).strip("_")
        bd = out / name
        bd.mkdir()

        def row(r):     # provenance travels with every row (prompt set: origin, source, licence, corpus build)
            return {"text": r["text"], "domain": "box", "doc_id": _doc_id(r["text"]), "probe_id": r["id"],
                    "topics": r["topics"], "source": r.get("group", ""), "origin": r.get("origin"),
                    "license": r.get("license"), "probe_source": r.get("source"), "method": r.get("method"),
                    "path_key": r.get("path_key"), "probe_sha256": r.get("sha256"),
                    "corpus_build": r.get("corpus_build") or build}
        val_all = [r for r in fit if h(r.get("group", r["id"]), a.seed + 7) % a.val_every == 0]
        vid = {r["id"] for r in val_all}
        carve = [r for r in fit if r["id"] not in vid]       # synthetic fit rows stay in carve (coordinator)
        val = [r for r in val_all if r.get("origin") != "synthetic"]   # ... but never in val: the export point is
                                                                       # chosen on the kind of text held is scored on
        held_all = [r for r in rs if r["split"] == "held"]
        held = [r for r in held_all if r.get("origin") != "synthetic"]   # never model-generated held text (coordinator)
        files = {"docs": carve, "val": val, "held": held}
        # minimum-count gate (review 2026-11-16): count what the carve will actually see
        lens = {r["id"]: ntok(r["text"]) for r in carve + val + held}
        ok_w = lambda rs: sum(lens[r["id"]] >= a.seq_len for r in rs)                     # noqa: E731
        ok_p = lambda rs, n: min(n, sum(lens[r["id"]] >= a.prompt_min for r in rs))       # noqa: E731
        counts = {"carve_windows": 2 * ok_w(carve), "val_windows": a.val_windows * ok_w(val) + 4 * ok_p(val, a.val_prompts),
                  "held_windows": ok_w(held) + 4 * ok_p(held, a.held_prompts)}
        short = [k for k, need in (("carve_windows", a.min_carve), ("val_windows", a.min_val),
                                   ("held_windows", a.min_held)) if counts[k] < need]
        if short:
            print(f"skip {b}: too few {short} {counts} (probes to extend)")
            extend.append({"node": b, **counts, "short": short})
            continue
        for k, v in files.items():
            (bd / f"{k}.jsonl").write_text("".join(json.dumps(row(r)) + "\n" for r in v))
        for k, n in (("carve", a.carve_prompts), ("val", a.val_prompts), ("held", a.held_prompts)):
            src = files["docs" if k == "carve" else k]
            pick = sorted((r for r in src if lens[r["id"]] >= a.prompt_min), key=lambda r: h(r["id"], a.seed + 11))[:n]
            (bd / f"prompts_{k}.jsonl").write_text("".join(json.dumps({**row(r), "box": name, "role": k}) + "\n"
                                                         for r in pick))
        boxes.append({"box": name, "node": b, "kind": "language" if b.startswith("lang.") else "subject",
                      "topics": sorted({r["topics"][0] for r in rs}), "carve_docs": len(carve), "val_docs": len(val),
                      "held_docs": len(held), "held_synthetic_dropped": len(held_all) - len(held),
                      "val_synthetic_dropped": len(val_all) - len(val), **counts,
                      "held_origins": dict(__import__("collections").Counter(r.get("origin") for r in held)),
                      **{f: sorted(t for t in {r["topics"][0] for r in rs} if t in ts) for f, ts in flags.items()}})
        print(name, len(carve), "carve", len(val), "val", len(held), "held")
    (out / "extend.json").write_text(json.dumps(extend, indent=1))
    (out / "boxes.json").write_text(json.dumps({"schema": "mlx2.amoeba-fn-boxset.v1", "corpus": a.corpus,
                                                "corpus_sha256_12": sha12(a.corpus), "seed": a.seed,
                                                "val_windows_per_doc": a.val_windows,
                                                "val_every": a.val_every, "boxes": boxes}, indent=1))


if __name__ == "__main__":
    main()
