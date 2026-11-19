"""Corpora for carving, re-encoded with the donor's own tokenizer.

Two sources:

  legacy      the 2026-11-13 amoeba corpus (core corpus v1: Qwen3-0.6B ids per domain). Decoded
              with the Qwen3-0.6B tokenizer and re-encoded with the donor tokenizer, with the same
              shuffles and split sizes as scripts/amoeba27b_lib.load_corpus
              (claude/amoeba-27b-80bc6a@37d75f144), so a 27B run here reproduces the 27B windows.
              (The 0.6B tokenizer only decodes text; Qwen3-0.6B is never a donor.)
  jsonl:PATH  one JSON object per line: {"text", "domain", "source"?, "split"?}. A row's "split"
              (fit / held / mean / spare / fresh) is honoured when present; otherwise rows are
              split by a hash of "source" (Pierre: topic splits are hashed by source file), so
              every row of one file lands in the same split. Corpus v2 and trace corpora (the
              GSM8K-train traces for the T3 residue) use this form.

Never carve on measurement sets (GSM8K test, the six-problem set, MMLU-Pro, GPQA): a jsonl row
whose source names one of them is refused.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

LEGACY = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/corpus-2026-11-13/corpus.json"
LEGACY_DOMAINS = ["tinystories_train", "repo_markdown", "donor_generated"]
MEASUREMENT_MARKERS = ("gsm8k-test", "gsm8k/test", "gsm8k_test", "six-problem", "mmlu-pro", "mmlu_pro", "gpqa")
SPLITS = ("held", "mean", "fit", "spare")


def sha12(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# Loss masks (FN calibration, coordinator 2026-11-16): an on-policy row "prompt + continuation" is carved with the loss
# on its continuation and the last few prompt tokens only. A sequence's first scored target index is kept here,
# keyed by the sequence's token tuple (sequences are plain id lists everywhere else); absent = every position.
LOSS_FROM: dict = {}
VAL: dict = {}        # {domain: [seqs]} from --val-file (export-point validation documents), filled by load()
SEQ_SRC: dict = {}    # windowed loader: token tuple -> [{"file", "row", "doc", "off", "len"}, ...]: the exact source row of
                      # every sequence (prompt set: Pierre 2026-11-16, a tile set ships with every text it used)
SEQ_DOC: dict = {}    # windowed loader: token tuple -> [(doc_id, token offset), ...] (every claimant; identical windows
                      # from different documents collide), so an atlas sequence maps to its source


CODE_RE = None


def cut_at_first_code(text):
    """FN calibration review 2026-11-16: code-corpus rows (Magicoder OSS-Instruct) open with an English task statement
    (35-47% of the characters). Keep the document from its first code line or fence; unchanged when none is found
    or when less than 200 characters would remain."""
    import re
    global CODE_RE
    if CODE_RE is None:
        CODE_RE = re.compile(r"^(```|#include|#!|\s{4,}\S|def |class |import |from \S+ import|package |using |"
                             r"public |private |fn |func |let |const |var |function |<\?php|struct |template)", re.M)
    m = CODE_RE.search(text)
    return text[m.start():] if m and m.start() < len(text) - 200 else text


def loss_from(seq) -> int:
    return LOSS_FROM.get(tuple(seq), 0)


def load(spec, host, seq_len=64, n_mean=240, n_fit=900, n_held=60, seed=0, log=print, windows=0, window_seed=0,
         strip_prefix=0.0, held_file=None, held_len=0, keep_prompt=32, val_file=None, code_cut=False, val_windows=0):
    """-> (split {domain: {held, mean, fit, spare: [ids]}}, receipt).

    windows / strip_prefix / held_file switch a jsonl corpus to the windowed loader (``_windowed``); without them the
    loader is the original one, sequence for sequence."""
    VAL.clear()
    if val_file and not spec.startswith("jsonl:"):
        raise SystemExit("--val-file needs a jsonl corpus (the windowed loader)")
    SEQ_DOC.clear()
    SEQ_SRC.clear()
    if spec.startswith("jsonl:") and (windows or strip_prefix or held_file or val_file or code_cut):
        p = Path(spec[len("jsonl:"):]).expanduser()
        split, rec = _windowed(p, host, seq_len, n_mean, n_fit, n_held, seed, log, windows, window_seed, strip_prefix,
                               held_file, held_len or seq_len, keep_prompt, val_file, code_cut, val_windows)
        return split, {"corpus": "jsonl", "path": str(p), "sha256_12": sha12(p), **rec}
    if spec in (None, "", "legacy"):
        return _legacy(host, seq_len, n_mean, n_fit, n_held, seed, log), {"corpus": "legacy", "path": str(LEGACY),
                                                                          "sha256_12": sha12(LEGACY)}
    if spec.startswith("jsonl:"):
        p = Path(spec[len("jsonl:"):]).expanduser()
        return _jsonl(p, host, seq_len, n_mean, n_fit, n_held, seed, log), {"corpus": "jsonl", "path": str(p),
                                                                            "sha256_12": sha12(p)}
    raise SystemExit(f"unknown corpus spec {spec!r} (legacy | jsonl:PATH)")


def _legacy(host, seq_len, n_mean, n_fit, n_held, seed, log):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))
    blob = json.loads(LEGACY.read_text())["seqs"]
    rng = random.Random(seed)
    split = {}
    for dn in LEGACY_DOMAINS:
        seqs = []
        for q in blob[dn]:
            ids = host.encode(src.decode(q))
            if len(ids) >= seq_len:
                seqs.append(ids[:seq_len])
        rng.shuffle(seqs)
        km, kf = n_mean // 3, n_fit // 3
        split[dn] = {"held": seqs[:n_held], "mean": seqs[n_held:n_held + km],
                     "fit": seqs[n_held + km:n_held + km + kf], "spare": seqs[n_held + km + kf:]}
        log(f"corpus {dn}: {len(seqs)} seqs of {seq_len} (held {len(split[dn]['held'])}, mean "
            f"{len(split[dn]['mean'])}, fit {len(split[dn]['fit'])}, spare {len(split[dn]['spare'])})")
    return split


def _bucket(source: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _jsonl(path, host, seq_len, n_mean, n_fit, n_held, seed, log):
    rows = [json.loads(x) for x in path.read_text().split("\n") if x.strip()]
    bad = [r.get("source", "") for r in rows if any(m in str(r.get("source", "")).lower() for m in MEASUREMENT_MARKERS)]
    if bad:
        raise SystemExit(f"corpus {path}: refusing measurement-set rows (never carve on them): {sorted(set(bad))[:5]}")
    by = {}
    for r in rows:
        by.setdefault(r.get("domain", "text"), []).append(r)
    doms = sorted(by)
    split = {}
    rng = random.Random(seed)
    for dn in doms:
        out = {k: [] for k in SPLITS}
        free = []
        for r in by[dn]:
            ids = host.encode(r["text"])
            if len(ids) < seq_len:
                continue
            ids = ids[:seq_len]
            sp = r.get("split")
            if sp == "fresh":
                sp = "held"
            if sp in out:
                out[sp].append(ids)
            else:
                free.append((_bucket(str(r.get("source", r["text"][:64])), seed), ids))
        free.sort(key=lambda t: t[0])
        km, kf = n_mean // len(doms), n_fit // len(doms)
        need = [("held", n_held), ("mean", km), ("fit", kf)]
        i = 0
        for name, k in need:
            take = max(0, k - len(out[name]))
            out[name] += [ids for _, ids in free[i:i + take]]
            i += take
        out["spare"] += [ids for _, ids in free[i:]]
        for k in SPLITS:
            rng.shuffle(out[k])
        split[dn] = out
        log(f"corpus {dn}: " + ", ".join(f"{k} {len(out[k])}" for k in SPLITS))
    return split


def domains(split):
    return list(split)


def shared_prefixes(texts, share):
    """The longest line-aligned (else sentence-aligned) prefix that more than `share` of `texts` start with, or None.
    Templates and system preambles (logic cells: one preamble in every row) are stripped before windowing."""
    from collections import Counter
    best = None
    for sep in ("\n", ". "):
        cnt = Counter()
        for t in texts:
            parts = t.split(sep)
            for k in range(1, min(12, len(parts))):
                cnt[sep.join(parts[:k]) + sep] += 1
        for pre, n in cnt.items():
            if n >= 2 and n > share * len(texts) and len(pre.strip()) >= 20 and (best is None or len(pre) > len(best[0])):
                best = (pre, n)
        if best:
            break
    return best


def _doc_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _windowed(path, host, seq_len, n_mean, n_fit, n_held, seed, log, windows, window_seed, strip_prefix, held_file,
              held_len, keep_prompt, val_file=None, code_cut=False, val_windows=0):
    """Windowed jsonl loader (FN calibration (a), (b), (d); coordinator 2026-11-16).

    (b) per domain, a shared template prefix (more than `strip_prefix` of the domain's rows) is stripped first.
    (a) a document row gives `windows` windows of seq_len tokens at seeded random offsets across the whole document
        (0 = its first seq_len tokens); offsets are recorded. An on-policy row (has "prompt") gives one sequence: the
        LAST seq_len tokens of prompt + continuation, scored from keep_prompt tokens before the continuation.
    Splits are by DOCUMENT (every window of a document, and every continuation of its prompt via "doc_id", shares
    its document's split; free documents are ordered by a seeded hash of the document, not of its source file, so
    one source never fills a split in file order). mean / fit are filled with whole documents.
    (d) held_file: its rows are the held split of every domain (cropped to held_len tokens, loss masks as above);
        corpus rows marked held are then not used at all, and a held document that also appears in the corpus
        (same doc_id) is refused."""
    import numpy as np
    rows = [json.loads(x) for x in path.read_text().split("\n") if x.strip()]
    bad = [r.get("source", "") for r in rows if any(m in str(r.get("source", "")).lower() for m in MEASUREMENT_MARKERS)]
    if bad:
        raise SystemExit(f"corpus {path}: refusing measurement-set rows (never carve on them): {sorted(set(bad))[:5]}")
    by = {}
    for r in rows:      # the document id is taken before any stripping (generations carry it as "doc_id")
        r2 = {**r, "_doc": r.get("doc_id") or _doc_id(r["text"])}
        if code_cut and r.get("prompt") is None and str(r.get("domain", "")).startswith("code"):
            r2["text"] = cut_at_first_code(r["text"])
            r2["_cut"] = len(r["text"]) - len(r2["text"])
        by.setdefault(r.get("domain", "text"), []).append(r2)
    doms = sorted(by)
    rec = {"loader": "windowed", "windows": windows, "window_seed": window_seed, "strip_prefix": strip_prefix,
           "keep_prompt": keep_prompt, "held_file": str(held_file) if held_file else None, "held_len": held_len,
           "code_cut": bool(code_cut), "val_windows": val_windows or windows,
           "stripped": {}, "offsets": {}, "counts": {}}
    split = {}

    def src(r, o, L):
        key = (f"gen:{r.get('prompt_key')}:{r.get('mode')}:{r.get('seed')}" if r.get("prompt") is not None
               else r.get("probe_id") or r["_doc"])
        return {"file": r.get("_file", "corpus"), "row": key, "doc": r["_doc"], "off": int(o), "len": int(L),
                "text_cut_chars": int(r.get("_cut", 0))}      # offsets are tokens of the text after this many chars

    def seqs_of(r, L, dn, nwin=None):
        nw = windows if nwin is None else nwin
        if r.get("prompt") is not None:                      # on-policy row: tail window, masked loss
            ids = host.encode(r["text"])
            n_prompt = len(host.encode(r["prompt"]))
            if len(ids) - n_prompt < 8 or len(ids) < L:
                return []
            cut = len(ids) - L
            q = ids[cut:]
            LOSS_FROM[tuple(q)] = max(1, n_prompt - keep_prompt - cut)
            SEQ_SRC.setdefault(tuple(q), []).append(src(r, cut, L))
            return [q]
        ids = host.encode(r["text"])
        if len(ids) < L:
            return []
        if not nw:
            SEQ_DOC.setdefault(tuple(ids[:L]), []).append((r["_doc"], 0))
            SEQ_SRC.setdefault(tuple(ids[:L]), []).append(src(r, 0, L))
            return [ids[:L]]
        span = len(ids) - L
        h = int.from_bytes(hashlib.sha256(f"{seed}:{window_seed}:{dn}:{r['_doc']}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(h)
        offs = sorted(set(int(o) for o in rng.integers(0, span + 1, size=min(nw, span + 1))))
        rec["offsets"].setdefault(dn, []).append([r["_doc"], offs])
        for o in offs:
            SEQ_DOC.setdefault(tuple(ids[o:o + L]), []).append((r["_doc"], o))
            SEQ_SRC.setdefault(tuple(ids[o:o + L]), []).append(src(r, o, L))
        return [ids[o:o + L] for o in offs]
    corpus_docs = set()
    doc_split = {}                       # document key -> split, shared by every domain (a continuation follows its doc)
    strip_pre = {}
    gen_doms = [dn for dn in doms if any(r.get("prompt") is not None for r in by[dn])]
    for dn in [d for d in doms if d not in gen_doms] + gen_doms:       # documents first, then their continuations
        rs = by[dn]
        if strip_prefix:
            sp = shared_prefixes([r["text"] for r in rs if r.get("prompt") is None], strip_prefix)
            if sp:
                pre, n = sp
                rs = [{**r, "text": r["text"][len(pre):], "_cut": r.get("_cut", 0) + len(pre)}
                      if r.get("prompt") is None and r["text"].startswith(pre) else r for r in rs]
                rec["stripped"][dn] = {"prefix": pre[:160], "chars": len(pre), "rows": n}
                strip_pre[dn] = pre
        docs = {}                                            # doc key -> [split, [seqs]]
        for r in rs:
            key = r["_doc"]
            corpus_docs.add(key)
            sq = seqs_of(r, seq_len, dn)
            if not sq:
                continue
            sp = r.get("split")
            sp = "held" if sp == "fresh" else sp
            d = docs.setdefault(key, [sp, []])
            d[0] = d[0] or sp
            d[1].extend(sq)
        out = {k: [] for k in SPLITS}
        free = []
        for key, (sp, sq) in docs.items():
            sp = doc_split.get(key, sp)
            if sp == "held" and held_file:
                doc_split[key] = "held"                      # its continuations stay out of fit too
                continue                                     # the fixed held set replaces corpus held documents
            if sp in out:
                out[sp].extend(sq)
                doc_split[key] = sp
            else:
                free.append((hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key, sq))
        free.sort(key=lambda t: t[0])
        km, kf = n_mean // len(doms), n_fit // len(doms)
        i = 0
        for name, k in [("held", n_held)] * (not held_file) + [("mean", km), ("fit", kf)]:
            while len(out[name]) < k and i < len(free):
                out[name].extend(free[i][2])
                doc_split[free[i][1]] = name
                i += 1
        for _, key, sq in free[i:]:
            out["spare"].extend(sq)
            doc_split[key] = "spare"
        rng = random.Random(seed)
        for k in SPLITS:
            rng.shuffle(out[k])
        split[dn] = out
    if held_file:
        hrows = [json.loads(x) for x in Path(held_file).read_text().split("\n") if x.strip()]
        leak = sorted({r.get("doc_id") or _doc_id(r["text"]) for r in hrows} & corpus_docs)
        if leak:
            raise SystemExit(f"held file {held_file}: {len(leak)} held documents also in the carve corpus")
        for r in hrows:
            dn = r.get("domain", "text")
            split.setdefault(dn, {k: [] for k in SPLITS})
            r = {**r, "_doc": r.get("doc_id") or _doc_id(r["text"]), "_file": "held"}
            if r.get("prompt") is not None:
                q = seqs_of(r, min(held_len, len(host.encode(r["text"]))), dn)
            else:
                ids = host.encode(r["text"])
                q = [ids[:held_len]] if len(ids) >= held_len else []
                for x in q:
                    SEQ_SRC.setdefault(tuple(x), []).append(src(r, 0, held_len))
            split[dn]["held"].extend(q)
        rec["held_sha256_12"] = sha12(held_file)
    if val_file:   # whole validation documents: windowed like the carve corpus (same seq_len), continuations masked
        vrows = [json.loads(x) for x in Path(val_file).read_text().split("\n") if x.strip()]
        vkeys = {r.get("doc_id") or _doc_id(r["text"]) for r in vrows}
        hkeys = set()
        if held_file:
            hkeys = {json.loads(x).get("doc_id") or _doc_id(json.loads(x)["text"])
                     for x in Path(held_file).read_text().split("\n") if x.strip()}
        if vkeys & corpus_docs or vkeys & hkeys:
            raise SystemExit(f"val file {val_file}: {len(vkeys & corpus_docs)} documents in the carve corpus, "
                             f"{len(vkeys & hkeys)} in the held file")
        for r in vrows:      # shaped like the held set (review 2026-11-16): windows and continuations cropped to held_len
            r = {**r, "_doc": r.get("doc_id") or _doc_id(r["text"]), "_file": "val"}
            dn = r.get("domain", "text")
            if code_cut and r.get("prompt") is None and str(dn).startswith("code"):
                r["text"] = cut_at_first_code(r["text"])
            pre = strip_pre.get(dn)
            if pre and r.get("prompt") is None and r["text"].startswith(pre):
                r["text"] = r["text"][len(pre):]
            if r.get("prompt") is not None:
                q = seqs_of(r, min(held_len, len(host.encode(r["text"]))), "val:" + dn)
            else:
                # val documents may take more windows than carve documents (coordinator 2026-11-16: thin val splits)
                q = seqs_of(r, held_len, "val:" + dn, val_windows or windows)   # offsets under "val:<domain>"
            VAL.setdefault(dn, []).extend(q)
        empty = [dn for dn in split if split[dn].get("fit") and not VAL.get(dn)]
        extra = [dn for dn in VAL if dn not in split]
        if not any(VAL.values()) or empty or extra:
            raise SystemExit(f"val file {val_file}: unusable (no val sequences: {not any(VAL.values())}; carve domains "
                             f"without val sequences: {empty}; val domains not in the corpus or held set: {extra}). An "
                             f"empty val split would pick the smallest export box")
        rec["val_file"] = str(val_file)
        rec["val_sha256_12"] = sha12(val_file)
        rec["val_counts"] = {dn: len(v) for dn, v in VAL.items()}
        log("val " + ", ".join(f"{dn} {len(v)}" for dn, v in VAL.items()))
    for dn in split:
        rec["counts"][dn] = {k: len(v) for k, v in split[dn].items()}
        log(f"corpus {dn}: " + ", ".join(f"{k} {len(split[dn][k])}" for k in SPLITS))
    return split, rec
