#!/usr/bin/env python3
"""Language-gating qualification test of the aMoEba BAU plan on Flash-Next, as served, the gate deciding by itself
(research, unqualified; Pierre 2026-11-20 "build it and queue it for the end of the queue").

BAU plan: the ten non-English language boxes and hum.religion are gated; every other tile (topic boxes, pairs with
topics, lang.eng, the ml / sci.* family tiles, the core) is one always-open core. Gated (scripts/amoeba_langgate_lib.py
tile_gates, OR rule):
  box:<L> (10)                        open with L
  box:hum.religion                    open with religion
  pair:<L1>__<L2>                     open with L1 or L2
  pair:<L>__hum.religion              open with L or religion
  group:langs / group:cjkv            open with any member language
  pair:<L>__<topic>                   --mixed-pairs language (default): open with L; core: always open
Religion + topic pairs are core. The gate (amoeba_langgate_lib.GatePolicy, thresholds in GateConfig and the receipt):
prefill = script runs first (Kana -> jpn; Han-only -> cmn + jpn; Arabic -> arb; Devanagari -> hin; Cyrillic -> rus;
Vietnamese letters -> vie), then fastText LID per segment (Latin-script fra/deu/spa/vie/swh), an "in <language>"
request cue, fail open (all ten) on an uncertain segment or an uncovered language, and the religion detector
(hashed-word naive Bayes fitted on fit rows, threshold on val rows); religion opens hin (--religion-couples).
Decode = every --check-every tokens, the same detection on the trailing --window-tokens of the output; a newly
detected gate opens from that forward on and stays open (hold-open). No early-layer religion probe exists (box-trip's
progress score needs a full forward with drive capture, and ranks religion only about fifth on religion prompts),
so the detector is text-only; the next-token predictor (nexttok.py) is not used: no FN language probe is fitted.

Teacher forced, per-token gates (research/amoeba/carve token_gate): row t of a request's gate stream gates the
forward whose input is token t at every MoE layer, the same as a KV-cached decode switching masks between steps, so
missed opens cost exactly what serving would pay. Closed tiles are masked (zero contribution, the drop rule).
Position 0 runs whole in every layer (the battery's sink convention, --sink-layers).

Arms: full (no mask), open (every gated tile open: must equal full), closed (every gated tile closed: worst case),
auto (BAU, the gate deciding), oracle (the true language / topic per token), and variants on --focus sets
(auto_nohin / oracle_nohin: religion without the hin coupling; oracle_nopairs / auto_nopairs: gated pair tiles
closed; oracle_han: the Han rule as oracle, cmn and jpn both).
Sets (held only; the battery's held specs and its document selection, seed 0): english, gsm8k, prog.python, the ten
languages, hum.religion, --topics; and mixed sets built from held2 rows (never fit): mixed.request (an English
instruction "Answer the following in <language>." then a held2 document of that language, prompt = the
instruction), mixed.switch (a held2 English document, then a held2 document of the language with no request:
the mid-answer switch the decode check must catch; prompt = the first --prompt-tokens), mixed.doc (a language
document, then English). --mixed-per-lang documents per language each.
Metrics: held dCE vs full per set (all targets, as the battery, and decode targets >= prompt), paired SE and the
source-group bootstrap (scripts/amoeba_battery_stats.py); gate accuracy against the oracle at prefill and per
decode row; missed-open (gate, row) pairs and their dCE; rows served fully gated; bytes of gated routed-expert
weights a request never opens (approximate and exact layout); greedy generations per language (native prompt,
English "answer in" request, and a switch prompt that names no language -- "then write it again in the main
language of <city>" -- so only decode re-detection can open it) with LID on the output, full vs auto.
Pre-registered rules (amoeba_langgate_lib.RULES) are written to OUT/prereg.json before anything is scored and
applied at the end (OUT/summary.json, summary.md); the summary is written whatever happens (finally), with every
cell not run listed, and a provisional summary is rewritten after every cell (a watchdog kill skips finally).
--time-cap-min bounds the scored work after the model loads; --stop-new-at HH:MM starts no new cell after that
local time (the night window).

--dry-run (CPU, no MLX, no model weights): the tokenizer from the model's tokenizer.json, the sets, every gate
stream, gate accuracy, memory accounting and the plan; writes the same OUT files minus the scored cells.

States: implemented; not qualified; not selected by any route; never observed-used in serving.
Usage (inside scripts/amoeba27b_window_q.sh via scripts/amoeba_fn_langgate_inner.sh):
  amoeba_fn_langgate.py --out DIR [--swap-growth-gb 8 --waiver-note "..."]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import amoeba_langgate_lib as G  # noqa: E402

SRC = Path.home() / "amoeba-runs/fn-catboxes-47214c256"
MODEL = Path.home() / "mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"
LID_DIR = Path.home() / "mlx-models/fasttext-lid-176"
TOPICS = ("sci.astronomy,life.cooking,hum.philosophy,hum.history,hum.linguistics,math.algebra,soc.law,sec.vuln,"
          "swe.devops,ml.inference,agent.tool_use,sci.medicine")
FOCUS = "arb,cmn,jpn,hin,hum.religion,mixed.switch"
NAME = {"fra": "French", "deu": "German", "spa": "Spanish", "cmn": "Chinese", "jpn": "Japanese", "rus": "Russian",
        "hin": "Hindi", "swh": "Swahili", "arb": "Arabic", "vie": "Vietnamese"}
NATIVE = {"fra": "Écris un court paragraphe de trois phrases sur la mer.",
          "deu": "Schreibe einen kurzen Absatz aus drei Sätzen über das Meer.",
          "spa": "Escribe un párrafo corto de tres frases sobre el mar.",
          "cmn": "请用三句话写一段关于大海的短文。",
          "jpn": "海について三つの文で短い段落を書いてください。",
          "rus": "Напиши короткий абзац из трёх предложений о море.",
          "hin": "समुद्र के बारे में तीन वाक्यों का एक छोटा अनुच्छेद लिखिए।",
          "swh": "Andika aya fupi ya sentensi tatu kuhusu bahari.",
          "arb": "اكتب فقرة قصيرة من ثلاث جمل عن البحر.",
          "vie": "Hãy viết một đoạn văn ngắn gồm ba câu về biển."}
CAPITAL = {"fra": "Paris", "deu": "Berlin", "spa": "Madrid", "cmn": "Beijing", "jpn": "Tokyo", "rus": "Moscow",
           "hin": "the Indian state of Uttar Pradesh", "swh": "Dar es Salaam", "arb": "Riyadh", "vie": "Hanoi"}
RECEIPT_SCHEMA = "mlx2.amoeba-fn-langgate.v1"


def git_rev():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], capture_output=True,
                          text=True).stdout.strip() or "unknown"


def sha12(path):
    return G.sha256_file(path)[:12]


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


# ----------------------------------------------------------------------------- held sets

def parse_spec(spec):
    """battery held spec NAME=[gsm8k:]PATH[?filter][@TOKENS] -> (name, kind, path, filter, T)."""
    name, _, rest = spec.partition("=")
    rest, _, tok = rest.partition("@")
    kind = "gsm8k" if rest.startswith("gsm8k:") else "corpus"
    path, _, filt = rest.removeprefix("gsm8k:").partition("?")
    return name, kind, path, filt, int(tok or 256)


SUBSTITUTED = {}


def resolve(path):
    """A held path; a missing snapshot copy of a repository benchmark file (.../benchmarks/<x>) resolves to this
    tree's copy (the fixed benchmark prompts), recorded in SUBSTITUTED with its sha256."""
    p = Path(path)
    if p.exists() or "/benchmarks/" not in str(p):
        return str(p)
    q = ROOT / "benchmarks" / str(p).split("/benchmarks/", 1)[1]
    if not q.exists():
        return str(p)
    SUBSTITUTED[str(p)] = {"used": str(q), "sha256_12": sha12(q)}
    return str(q)


def pick_docs(rows, encode, T, n, seed):
    """The battery's selection: seeded shuffle, the first n documents of >= T tokens, cut to T."""
    rows = list(rows)
    np.random.default_rng(seed).shuffle(rows)
    seqs, used = [], []
    for r in rows:
        t = encode(r["text"])
        if len(t) >= T:
            seqs.append(t[:T])
            used.append(r)
        if len(seqs) >= n:
            break
    return seqs, used


class HeldSet:
    def __init__(self, name, kind, T, docs, rows, prompts, labels, spec, request=None):
        self.name, self.kind, self.T = name, kind, T
        self.docs, self.rows, self.prompts, self.labels, self.spec = docs, rows, prompts, labels, spec
        self.request = request or [None] * len(docs)


def kind_of(name, topics):
    if name == "english":
        return "english"
    if name == "gsm8k":
        return "gsm8k"
    if name == "prog.python":
        return "code"
    if name in G.LANG10:
        return "lang"
    if name == G.RELIGION:
        return "religion"
    if name in topics:
        return "topic"
    raise ValueError(name)


def _val_spec(name, kind, path, prep):
    """--split val (tuning): the set's val rows: its held file's val split, else its box's carve corpus val split,
    GSM8K's train file; None when no val rows exist (the set is skipped in tuning)."""
    if kind == "gsm8k":
        q = ROOT / "benchmarks" / "gsm8k" / "train.jsonl"
        return ("gsm8k", str(q), "") if q.exists() else None
    for p in (path, ((prep or {}).get("boxes") or {}).get(name, {}).get("target_corpus")):
        if p and Path(p).exists() and '"split": "val"' in Path(p).read_text():
            return ("corpus", p, "split=val")
    return None


def build_sets(meta, encode, a, log=print, prep=None):
    """Every held set (battery held specs, the battery's documents) and the mixed sets (held2 rows); --split val
    builds the same sets from val rows (tuning)."""
    import amoeba_box_battery as B
    val = getattr(a, "split", "held") == "val"
    split2 = "split=val" if val else "split=held2"
    topics = [x for x in a.topics.split(",") if x]
    want = ["english", "gsm8k", "prog.python", *G.LANG10, G.RELIGION, *topics]
    specs = {k: v["spec"] for k, v in meta["held"].items()}
    miss = [k for k in want if k not in specs]
    if miss:
        raise SystemExit(f"battery meta has no held spec for {miss}")
    out = {}
    P = a.prompt_tokens
    for name in want:
        _, kind, path, filt, T = parse_spec(specs[name])
        path = resolve(path)
        if val:
            v = _val_spec(name, kind, path, prep)
            if v is None:
                log(f"val {name}: no val rows (skipped in tuning)")
                continue
            kind, path, filt = v
        rows = B.read_held_rows(path, kind, filt)
        if kind == "gsm8k":
            rows = [{**r, "id": f"gsm8k:{i}"} for i, r in enumerate(rows)]
        seqs, used = pick_docs(rows, encode, T, a.docs, a.seed)
        if len(seqs) < 4:
            raise SystemExit(f"held {name}: {len(seqs)} documents of {T} tokens")
        k = kind_of(name, topics)
        lab = frozenset([name]) if k in ("lang", "religion") else frozenset()
        out[name] = HeldSet(name, k, T, seqs, used, [min(P, T // 2)] * len(seqs), [[lab] * T for _ in seqs],
                            specs[name])
        log(f"held {name}: {len(seqs)} x {T}")
    T = a.tokens
    _, _, epath, _, _ = parse_spec(specs["english"])
    eng = [r for r in B.read_held_rows(epath, "corpus", split2)]
    eseqs, erows = pick_docs(eng, encode, T, 10 * a.mixed_per_lang, a.seed + 1)
    mixed = {"mixed.request": [], "mixed.switch": [], "mixed.doc": []}
    sep = encode("\n\n")
    for li, L in enumerate(G.LANG10):
        _, _, lpath, _, _ = parse_spec(specs[L])
        lrows = B.read_held_rows(lpath, "corpus", split2)
        lseqs, lused = pick_docs(lrows, encode, T, 3 * a.mixed_per_lang, a.seed + 1)
        if len(lseqs) < 3 * a.mixed_per_lang:
            raise SystemExit(f"{L}: {len(lseqs)} held2 documents of {T} tokens (need {3 * a.mixed_per_lang})")
        def meta_row(r, what, L=L, eng=None):
            row = {"id": f"{what}:{L}:{r.get('id')}", "group": f"{L}:{r.get('group') or r.get('id')}", "register": L,
                   "source_ids": [str(r.get("id"))] + ([str(eng.get("id"))] if eng else []),
                   "text": r.get("text", "")}
            if eng:
                row["text_english"] = eng.get("text", "")
            return row
        for j in range(a.mixed_per_lang):
            # request: English instruction, then the language document (prompt = the instruction)
            instr = f"Instruction: Answer the following in {NAME[L]}.\n\nAnswer:\n"
            it = encode(instr)
            if len(it) >= T // 2:
                raise SystemExit(f"--tokens {T}: the request instruction takes {len(it)} tokens (need < T / 2)")
            d = lseqs[j]
            seq = (it + d)[:T]
            mixed["mixed.request"].append((seq, len(it), [frozenset()] * len(it) + [frozenset([L])] * (T - len(it)),
                                           meta_row(lused[j], "request"), instr))
            # switch: English document, then the language (no request; prompt inside the English part)
            ei = (li * a.mixed_per_lang + j) % len(eseqs)
            e = eseqs[ei]
            h = T // 2
            d2 = lseqs[a.mixed_per_lang + j]
            seq = (e[:h] + sep + d2)[:T]
            k = h + len(sep)
            mixed["mixed.switch"].append((seq, min(a.prompt_tokens, h),
                                          [frozenset()] * k + [frozenset([L])] * (T - k),
                                          meta_row(lused[a.mixed_per_lang + j], "switch", eng=erows[ei]), None))
            # doc: the language, then English
            d3 = lseqs[2 * a.mixed_per_lang + j]
            seq = (d3[:h] + sep + e)[:T]
            mixed["mixed.doc"].append((seq, min(a.prompt_tokens, h),
                                       [frozenset([L])] * h + [frozenset()] * (T - h),
                                       meta_row(lused[2 * a.mixed_per_lang + j], "doc", eng=erows[ei]), None))
    for name, items in mixed.items():                      # language-major order: deterministic
        out[name] = HeldSet(name, "mixed", T, [x[0] for x in items], [x[3] for x in items], [x[1] for x in items],
                            [x[2] for x in items], f"built from held2 rows ({name})", [x[4] for x in items])
        log(f"{name}: {len(items)} x {T}")
    return out


# ----------------------------------------------------------------------------- religion detector

def fit_religion(prep, meta, a, held_sets, log=print, purge_val=True):
    """ReligionNB on fit rows (positives: hum.religion's carve corpus; negatives: up to --rel-neg-per-box fit rows of
    every other box outside its lineage and of English), threshold on val rows; every fit / val row that shares an
    id, a sha256 or a source group with any held row of any scored set is dropped first (purge check)."""
    import amoeba_box_battery as B  # noqa: F401
    rng = np.random.default_rng(a.seed + 5)
    held_ids, held_text = set(), set()
    for s in held_sets.values():
        for r in s.rows:
            held_ids.add(str(r.get("id")))
            held_ids.update(r.get("source_ids") or [])
            for k in ("text", "text_english"):
                if r.get(k):
                    held_text.add(hashlib.sha256(r[k].encode()).hexdigest())
    rel_groups = {str(r["group"]) for r in held_sets[G.RELIGION].rows if r.get("group")} if G.RELIGION in held_sets \
        else set()

    def clean(rows, groups=()):
        return [r for r in rows if str(r.get("id")) not in held_ids and str(r.get("group")) not in groups
                and hashlib.sha256(r.get("text", "").encode()).hexdigest() not in held_text]

    def rows_of(path, split):
        rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
        return [r for r in rows if r.get("split", "fit") == split]
    boxes = prep["boxes"]
    pos_path = boxes[G.RELIGION]["target_corpus"]
    pf, pv = rows_of(pos_path, "fit"), rows_of(pos_path, "val")
    pos_fit = clean(pf, rel_groups)
    pos_val = clean(pv, rel_groups) if purge_val else pv
    pos_dropped = len(pf) + len(pv) - len(pos_fit) - len(pos_val)
    lineage = {G.RELIGION} | set(((prep.get("nested") or {}).get("ancestors") or {}).get(G.RELIGION, []))
    neg_fit, neg_val, srcs = [], [], []
    paths = sorted({v["target_corpus"] for b, v in boxes.items() if b not in lineage and v.get("target_corpus")}
                   | {boxes[G.RELIGION]["baseline_corpus"]})
    dropped = 0
    for p in paths:
        if not Path(p).exists():
            continue
        f, v = rows_of(p, "fit"), rows_of(p, "val")
        n0 = len(f) + len(v)
        f, v = clean(f), (clean(v) if purge_val else v)
        dropped += n0 - len(f) - len(v)
        rng.shuffle(f)
        rng.shuffle(v)
        neg_fit += f[:a.rel_neg_per_box]
        neg_val += v[:max(1, a.rel_neg_per_box // 4)]
        srcs.append(p)
    W = a.rel_window_words
    nb = G.ReligionNB().fit([r["text"] for r in pos_fit], [r["text"] for r in neg_fit])
    win = lambda rows: [G.word_window(r["text"], W) for r in rows if r.get("text")]  # noqa: E731
    m = nb.calibrate(win(pos_val), win(neg_val), a.rel_recall, a.rel_recall_decode)
    nb.meta.update(positive_corpus=pos_path, negative_corpora=len(srcs), window_words=W,
                   purged_rows=dropped, purged_positive_rows=pos_dropped,
                   purge="fit / val rows sharing an id or a text sha256 with any scored held row are dropped; "
                         "religion rows also when they share a source group with a religion held row",
                   method="hashed word uni+bigram multinomial NB, mean log-ratio per feature; "
                   "threshold = (1 - recall) quantile of val positives on prompt-sized windows",
                   held_rows_read=0)
    log(f"religion NB: fit {len(pos_fit)}+/{len(neg_fit)}-, val recall {m['val_recall']} fpr {m['val_fpr']} "
        f"threshold {m['threshold']}")
    if not pos_fit or not pos_val:
        raise SystemExit("religion detector: no fit or val positives left after the purge (refusing)")
    return nb


# ----------------------------------------------------------------------------- gate streams

def arm_plan(focus):
    """{arm: (stream source, groups variant, sets)}: sets None = every set."""
    return {"full": (None, None, None), "auto": ("auto", "or", None), "oracle": ("oracle", "or", None),
            "open": ("ones", "or", None), "closed": ("zeros", "or", None),
            "auto_nohin": ("auto_nohin", "or", focus), "oracle_nohin": ("oracle_nohin", "or", focus),
            "oracle_nopairs": ("oracle", "nopairs", focus), "auto_nopairs": ("auto", "nopairs", focus),
            "oracle_han": ("oracle_han", "or", [x for x in focus if x in ("cmn", "jpn")])}


PHASES = (("full", "auto"), ("oracle",), ("open", "closed"), ("generations",),
          ("auto_nohin", "oracle_nohin", "oracle_nopairs", "auto_nopairs", "oracle_han"))


def streams_for(sets, policies, couples, decode, focus=None, log=print):
    """Every gate stream: {source: {set: [stream [T, gates]]}} and the auto logs (policies other than auto: only
    on the focus sets when focus is given)."""
    out = {k: {} for k in ("auto", "auto_nohin", "oracle", "oracle_nohin", "oracle_han", "ones", "zeros")}
    logs = {}
    t0 = time.time()
    for name, s in sets.items():
        for k, pol in policies.items():
            if k != "auto" and focus is not None and name not in focus:
                continue
            res = [G.simulate(d, P, decode, pol, req) for d, P, req in zip(s.docs, s.prompts, s.request)]
            out[k][name] = [r[0] for r in res]
            if k == "auto":
                logs[name] = [r[1] for r in res]
        out["oracle"][name] = [G.oracle_stream(lb, couples) for lb in s.labels]
        out["oracle_nohin"][name] = [G.oracle_stream(lb, ()) for lb in s.labels]
        han = [[(x | {"cmn", "jpn"}) if x & {"cmn", "jpn"} else x for x in lb] for lb in s.labels]
        out["oracle_han"][name] = [G.oracle_stream(lb, couples) for lb in han]
        out["ones"][name] = [np.ones((s.T, len(G.GATES)), bool) for _ in s.docs]
        out["zeros"][name] = [np.zeros((s.T, len(G.GATES)), bool) for _ in s.docs]
    log(f"gate streams for {sum(len(s.docs) for s in sets.values())} requests in {time.time() - t0:.0f}s")
    return out, logs


def gate_report(sets, streams, logs, unit_idx, groups, W, cfg, mem_args):
    """Per set: gate accuracy (auto vs oracle), missed opens, fully gated rows, bytes never opened per request."""
    rep, pool = {}, {"needed": 0, "missed": 0, "by_kind": {}}
    cache = {}
    for name, s in sets.items():
        ms = [G.gate_metrics(a_, o_, P) for a_, o_, P in zip(streams["auto"][name], streams["oracle"][name],
                                                               s.prompts)]
        never = []
        for st in streams["auto"][name]:
            opened = {g for g, v in G.group_streams(st, groups).items() if v.any()}
            closed = tuple(sorted(k for g, ks in groups.items() if g and g not in opened for k in ks))
            if closed not in cache:
                cache[closed] = G.gated_bytes(unit_idx, W, closed, **mem_args)
            never.append(cache[closed])
        need, miss = sum(m["needed"] for m in ms), sum(m["missed"] for m in ms)
        r = {"kind": s.kind, "requests": len(ms),
             "prefill_exact_rate": float(np.mean([m["prefill_exact"] for m in ms])),
             "prefill_superset_rate": float(np.mean([m["prefill_superset"] for m in ms])),
             "needed": need, "missed": miss, "missed_rate": (miss / need) if need else None,
             "needed_decode": sum(m["needed_decode"] for m in ms), "missed_decode": sum(m["missed_decode"] for m in ms),
             "extra_rate": float(sum(m["extra"] for m in ms) / max(1, sum(m["rows"] for m in ms) * len(G.GATES))),
             "rows_fully_gated_share": float(sum(m["rows_fully_gated"] for m in ms) / max(1, sum(m["rows"] for m in ms))),
             "row_exact_share": float(sum(m["row_exact"] for m in ms) / max(1, sum(m["rows"] for m in ms))),
             "missed_by_gate": {}, "prefill_opened_counts": {},
             "never_opened_bytes_mean": {k: float(np.mean([x[k] for x in never])) for k in ("approx_bytes",
                                                                                         "exact_bytes", "units")},
             "fail_open_requests": sum(any(str(w).startswith("fail_open") for v in lg["prefill"]["reasons"].values()
                                           for w in v) for lg in logs[name]),
             "decode_open_events": sum(len(lg["events"]) for lg in logs[name])}
        for m in ms:
            for g, v in m["missed_by_gate"].items():
                r["missed_by_gate"][g] = r["missed_by_gate"].get(g, 0) + v
        for lg in logs[name]:
            for g in lg["prefill"]["open"]:
                r["prefill_opened_counts"][g] = r["prefill_opened_counts"].get(g, 0) + 1
        rep[name] = r
        if s.kind in ("lang", "religion", "mixed"):
            pool["needed"] += need
            pool["missed"] += miss
            pk = pool["by_kind"].setdefault(s.kind, {"needed": 0, "missed": 0})
            pk["needed"] += need
            pk["missed"] += miss
    pool["rate"] = pool["missed"] / pool["needed"] if pool["needed"] else None
    return rep, pool


def memory_table(unit_idx, groups, kinds, W, mem_args, total_units):
    """Gated bytes by kind and in all, approximate and exact layout."""
    gated = [k for g, ks in groups.items() for k in ks]
    by = {}
    for k in gated:
        by.setdefault(kinds[k], []).append(k)
    out = {"per_kind": {kd: G.gated_bytes(unit_idx, W, ks, **mem_args) for kd, ks in sorted(by.items())},
           "all_gated": G.gated_bytes(unit_idx, W, gated, **mem_args), "units_total": int(total_units),
           "unit_bytes": dict(zip(("unit", "gate_up_rows"), G.unit_bytes(**mem_args))), **mem_args}
    out["all_gated"]["share_of_units"] = out["all_gated"]["units"] / total_units
    return out


# ----------------------------------------------------------------------------- scoring (model)

class Scorer:
    """Per-token CE (targets 2..T-1, the battery's text_ce positions) under BAU core + per-token gated groups."""

    def __init__(self, host, tz, core_keys, groups, sink, bs, log):
        import mlx.core as mx  # noqa: F401
        from research.amoeba.carve import __main__ as D
        self.host, self.tz, self.groups, self.bs, self.log, self.D = host, tz, groups, bs, log, D
        self.layers = tz.layers
        self.W = max(host.widths)
        if tuple(tz.shape) != (len(self.layers), self.W):
            raise SystemExit(f"tile shape {tz.shape} != [moe layers, {self.W}]")
        self.core = tz.mask(*core_keys).reshape(-1)
        self.gidx = {g: np.concatenate([tz.idx[k] for k in ks]).astype(np.int64) for g, ks in groups.items()}
        self.sink = host.L if sink < 0 else sink
        self._cache = {}

    def _dense(self, bits, base_ones):
        m = np.ones((self.host.L, self.W), np.float32) if base_ones else np.zeros((self.host.L, self.W), np.float32)
        m[self.layers] = bits.reshape(len(self.layers), self.W).astype(np.float32)
        return self.D.from_np(m, self.host)

    def masks(self, folded, merged):
        """(core device mask, [(device box mask, key)]): the core with every always-open group folded in."""
        key = tuple(sorted(tuple(sorted(g)) for g in folded))
        if key not in self._cache:
            c = self.core.copy()
            for g in folded:
                c[self.gidx[g]] = True
            if len(self._cache) > 24:                    # 63 MB each (float32 [48, 327680])
                self._cache.clear()
            self._cache[key] = self._dense(c, True)
        boxes = []
        for gs in merged:
            bk = ("box",) + tuple(sorted(tuple(sorted(g)) for g in gs))
            if bk not in self._cache:
                b = np.zeros(self.core.size, bool)
                for g in gs:
                    b[self.gidx[g]] = True
                self._cache[bk] = self._dense(b, False)
            boxes.append(self._cache[bk])
        return self._cache[key], boxes

    def configure(self, gstreams):
        """gstreams: {group: [B, T] bool}. Folds all-open groups into the core, drops all-closed ones and merges
        groups with identical streams into one box with that token gate."""
        import mlx.core as mx
        folded, merged = [], {}
        for g, st in gstreams.items():
            if not g or not st.any():
                continue
            if st.all():
                folded.append(g)
            else:
                merged.setdefault(st.tobytes(), (st, []))[1].append(g)
        core, boxes = self.masks(folded, [v[1] for v in merged.values()])
        c = self.host.ctl
        c.reset()
        c.core, c.sink_layers = core, self.sink
        c.boxes = [{"name": "+".join("|".join(sorted(g)) for g in gs), "mask": m,
                    "token_gate": mx.array(st.astype(np.float32))[..., None]}
                   for (st, gs), m in zip(merged.values(), boxes)]
        return len(folded), len(boxes)

    def ce(self, docs, streams=None):
        """[n, T - 2] float32 per-token CE; streams None = the full model (Control reset)."""
        import mlx.core as mx
        import amoeba_nexttok_fn_window as NW
        from research.amoeba.carve.carving import ce_rows
        out = []
        for i in range(0, len(docs), self.bs):
            ids = mx.array(docs[i:i + self.bs])
            T = ids.shape[1]
            if streams is None:
                self.host.ctl.reset()
            else:
                per = [G.group_streams(x, self.groups) for x in streams[i:i + self.bs]]
                self.configure({g: np.stack([v[g] for v in per]) for g in self.groups})
            lg = NW.logits_at(self.host, ids, list(range(1, T - 1)))
            ce = ce_rows(lg, ids[:, 2:])
            mx.eval(ce)
            del lg
            out.append(np.array(ce, dtype=np.float32))
            self.host.ctl.reset()
            mx.clear_cache()
        return np.concatenate(out)

    def set_state(self, open_groups):
        """Generation: the core plus the open groups as one static mask (a decode step's whole gate)."""
        c = self.host.ctl
        core, _ = self.masks([g for g in open_groups if g], [])
        c.reset()
        c.core, c.sink_layers = core, self.sink


def doc_stats(ce_arm, ce_full, prompts, groups_, strata):
    """Paired per-document dCE (all targets, decode targets) with SE and the group bootstrap."""
    import amoeba_battery_stats as ST
    a_all, f_all, a_dec, f_dec = [], [], [], []
    for x, y, P in zip(ce_arm, ce_full, prompts):
        a_all.append(float(x.mean()))
        f_all.append(float(y.mean()))
        k = max(0, P - 2)                                   # targets >= P (index j - 2)
        a_dec.append(float(x[k:].mean()))
        f_dec.append(float(y[k:].mean()))
    return {"all": ST.paired(a_all, f_all, groups_, strata), "decode": ST.paired(a_dec, f_dec, groups_, strata)}


# ----------------------------------------------------------------------------- generations

def gen_prompts():
    out = []
    for L in G.LANG10:
        out.append({"id": f"native:{L}", "kind": "native", "lang": L, "text": NATIVE[L]})
        out.append({"id": f"request:{L}", "kind": "request", "lang": L,
                    "text": f"Answer in {NAME[L]}: describe a busy market in three sentences."})
        out.append({"id": f"switch:{L}", "kind": "switch", "lang": L,       # no language name: no request cue
                    "text": f"Write one sentence in English about the mountains, then write the same sentence again "
                            f"in the main language of {CAPITAL[L]}."})
    out += [{"id": "english:sea", "kind": "english", "lang": "eng", "text": "Write a short paragraph about the sea."},
            {"id": "english:recipe", "kind": "english", "lang": "eng",
             "text": "Give a short recipe for tomato soup in three steps."},
            {"id": "religion:buddhism", "kind": "religion", "lang": "eng",
             "text": "Explain in a short paragraph what the Eightfold Path is in Buddhism."},
            {"id": "religion:exodus", "kind": "religion", "lang": "eng",
             "text": "Summarize the story of the Exodus in a short paragraph."}]
    return out


def strip_think(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def output_correct(lid, pr, text, min_p=0.5):
    """LID on the output: native / request / english / religion: the top label is the requested language; switch:
    some segment is the target language at >= min_p."""
    t = strip_think(text)
    want = G.LID_LABEL[pr["lang"]]
    top = lid.top(t.replace("\n", " "), 3) if t else []
    if pr["kind"] == "switch":
        segs = G.segments(t, 300)
        hit = any(dict(lid.top(s, 3)).get(want, 0.0) >= min_p for s in segs) if segs else False
        return bool(hit), top
    return bool(top and top[0][0] == want), top


def generate(host, scorer, pr, arm, policy, eos, n, decode):
    """Greedy, KV cache. auto: prefill gate from the user text, then re-detection every check_every tokens on the
    trailing window of the output, hold-open (the masks switch between steps)."""
    import mlx.core as mx
    sys.path.insert(0, str(ROOT / "scripts"))
    import amoeba_fn_box_trip as BT
    ids, route = BT.render(host, {"messages": [{"role": "user", "content": pr["text"]}]}, False)
    cfg = policy.cfg
    events = []
    opened = set()
    if arm == "auto":
        g0, info = policy.detect(pr["text"])
        opened |= g0
        events.append({"t": 0, "open": sorted(g0), "reasons": info["reasons"], "flags": info["flags"]})
    groups = scorer.groups

    def apply():
        if arm == "full":
            host.ctl.reset()
        else:
            scorer.set_state([g for g in groups if g and (set(g) & opened)])
    apply()
    out = []
    mk = getattr(host.model, "make_cache", None)
    cache = mk() if callable(mk) else None
    t0 = time.time()
    if cache is not None:
        lg = host.model(mx.array([ids], dtype=mx.int32), cache=cache)
    else:
        lg = host.full_logits(mx.array([ids], dtype=mx.int32))
    for step in range(n):
        t = int(mx.argmax(lg[0, -1].astype(mx.float32)).item())
        out.append(t)
        if t in eos:
            break
        full = step % cfg.check_every == cfg.check_every - 1
        if arm == "auto" and (full or (cfg.script_every > 0 and step % cfg.script_every == cfg.script_every - 1)):
            gs, inf = policy.detect(decode(out[-cfg.window_tokens:]), stage="decode", full=full)
            new = gs - opened
            if new:
                opened |= new
                events.append({"t": step + 1, "open": sorted(new), "reasons": {g: inf["reasons"][g] for g in new}})
                apply()
        if cache is not None:
            lg = host.model(mx.array([[t]], dtype=mx.int32), cache=cache)
        else:
            apply()
            lg = host.full_logits(mx.array([ids + out], dtype=mx.int32))
    host.ctl.reset()
    mx.clear_cache()
    return {"tokens": len(out), "text": decode([x for x in out if x not in eos]), "events": events,
            "route": route, "cache": cache is not None, "wall_s": round(time.time() - t0, 1),
            "opened": sorted(opened)}


# ----------------------------------------------------------------------------- main

def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--src", default=str(SRC), help="catboxes run: partition/, battery/battery.meta.json")
    ap.add_argument("--partition", default="", help="default SRC/partition")
    ap.add_argument("--battery-meta", default="", help="default SRC/battery/battery.meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lid-dir", default=str(LID_DIR))
    ap.add_argument("--docs", type=int, default=80)
    ap.add_argument("--tokens", type=int, default=256, help="mixed documents' length")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--mixed-per-lang", type=int, default=8)
    ap.add_argument("--topics", default=TOPICS)
    ap.add_argument("--focus", default=FOCUS, help="sets of the variant arms")
    ap.add_argument("--mixed-pairs", choices=("language", "core"), default="language",
                    help="pair tiles of a language and a topic: open with the language (default) or always open")
    ap.add_argument("--religion-couples", default="hin", help="gates religion also opens (comma list; '' = none)")
    ap.add_argument("--check-every", type=int, default=4)
    ap.add_argument("--window-tokens", type=int, default=64)
    ap.add_argument("--script-every", type=int, default=1, help="script-only re-detection every N tokens (0 = off)")
    ap.add_argument("--hold-tokens", type=int, default=-1)
    ap.add_argument("--lid-open-p", type=float, default=0.5)
    ap.add_argument("--uncertain-p", type=float, default=0.4)
    ap.add_argument("--uncertain-rule", choices=("gated_mass", "top1"), default="gated_mass")
    ap.add_argument("--uncertain-mass", type=float, default=0.2)
    ap.add_argument("--uncertain-min-letters", type=int, default=40)
    ap.add_argument("--lid-min-letters", type=int, default=12)
    ap.add_argument("--split", choices=("held", "val"), default="held",
                    help="val: tuning sets from val rows (dry run only; thresholds are tuned on these, never on held)")
    ap.add_argument("--on-uncertain", choices=("open_all", "none"), default="open_all")
    ap.add_argument("--rel-recall", type=float, default=0.95)
    ap.add_argument("--rel-recall-decode", type=float, default=0.8)
    ap.add_argument("--rel-window-words", type=int, default=48)
    ap.add_argument("--rel-neg-per-box", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce-batch", type=int, default=4)
    ap.add_argument("--sink-layers", type=int, default=-1, help="-1 = every layer (the battery's convention)")
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--gen-arms", default="full,auto")
    ap.add_argument("--time-cap-min", type=float, default=130.0, help="scored work after the model loads")
    ap.add_argument("--stop-new-at", default="", help="HH:MM local: start no new cell after this clock time "
                    "(the next occurrence after the start; the night window ends 07:00)")
    ap.add_argument("--memory-gb", type=float, default=82)
    ap.add_argument("--watchdog-gb", type=float, default=86)
    ap.add_argument("--watchdog-hard-gb", type=float, default=None)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--min-avail-gb", type=float, default=15.0)
    ap.add_argument("--compressor-growth-gb", type=float, default=12.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--wired-gb", default=os.environ.get("AMOEBA_WIRED_GB", "").strip() or "auto")
    ap.add_argument("--assert-ple", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: sets, gates, memory, plan (no model weights)")
    ap.add_argument("--skip-lid-sha", action="store_true", help="tests only")
    return ap


class TokJson:
    """The model's tokenizer.json on the CPU (dry run): encode without special tokens, decode."""

    def __init__(self, model):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(Path(model) / "tokenizer.json"))

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.t.decode(list(ids), skip_special_tokens=False)


def _sigterm(signum, frame):
    raise SystemExit(f"signal {signum}")


def main(argv=None, host_factory=None, lid_factory=None):
    a = parser().parse_args(argv)
    if argv is None:
        import signal
        signal.signal(signal.SIGTERM, _sigterm)         # SIGTERM runs finally (the summary), like an exception
    t_start = time.time()
    if a.stop_new_at and not a.dry_run and clock_deadline(a.stop_new_at, t_start) <= t_start:
        # the cutoff has already passed: every cell would be skipped after a full model load. Refuse before OUT
        # is created (the inner script refuses a non-empty OUT on the retry) and yield (75).
        print(f"langgate: --stop-new-at {a.stop_new_at} has just passed at {time.strftime('%H:%M:%S')}; not started",
              flush=True)
        raise SystemExit(75)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fh = open(out / "langgate.log", "a")

    def log(*m):
        s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in m)
        print(s, flush=True)
        log_fh.write(s + "\n")
        log_fh.flush()
    src = Path(a.src)
    pd = Path(a.partition or src / "partition")
    meta_p = Path(a.battery_meta or src / "battery" / "battery.meta.json")
    prep = json.loads((pd / "partition.json").read_text())
    meta = json.loads(meta_p.read_text())
    import amoeba_box_battery as B
    tz = B.Tiles(pd / "tiles.npz")
    couples = tuple(x for x in a.religion_couples.split(",") if x)
    if a.hold_tokens > 0 and "auto" in a.gen_arms.split(",") and not a.dry_run:
        raise SystemExit("--hold-tokens > 0: generations never close a gate (only -1 is implemented there)")
    if a.split == "val" and not a.dry_run:
        raise SystemExit("--split val is for CPU tuning (--dry-run) only")
    cfg = G.GateConfig(lid_open_p=a.lid_open_p, uncertain_p=a.uncertain_p, on_uncertain=a.on_uncertain,
                       uncertain_rule=a.uncertain_rule, uncertain_mass=a.uncertain_mass,
                       uncertain_min_letters=a.uncertain_min_letters, lid_min_letters=a.lid_min_letters,
                       religion_couples=couples, check_every=a.check_every, window_tokens=a.window_tokens,
                       script_every=a.script_every,
                       hold_tokens=a.hold_tokens)
    core_keys, groups, kinds = G.gate_groups(tz.keys, a.mixed_pairs)
    _, groups_np, _ = G.gate_groups(tz.keys, a.mixed_pairs, pairs="closed")
    W = tz.shape[1]
    mcfg = json.loads((Path(a.model) / "config.json").read_text())
    tcfg = mcfg.get("text_config", mcfg)
    q = mcfg.get("quantization") or {}
    mem_args = {"d": int(tcfg.get("hidden_size", 2560)), "bits": int(q.get("bits", 4)),
                "group_size": int(q.get("group_size", 64))}
    mem = memory_table(tz.idx, groups, kinds, W, mem_args, tz.size)
    focus = [x for x in a.focus.split(",") if x]
    plan = arm_plan(focus)
    receipt = {"schema": RECEIPT_SCHEMA, "status": "started", "dry_run": a.dry_run,
               "mlx2_source_revision": git_rev(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "checkpoint": prep.get("checkpoint"), "fingerprint": prep.get("fingerprint"),
               "partition": str(pd), "partition_json_sha256_12": sha12(pd / "partition.json"),
               "battery_meta": str(meta_p), "argv": sys.argv if argv is None else argv,
               "plan": {"bau": "gate the ten language boxes and hum.religion; every other tile is an always-open core",
                        "mixed_pairs": a.mixed_pairs, "religion_couples": list(couples),
                        "tile_kinds": {kd: sum(1 for v in kinds.values() if v == kd) for kd in sorted(set(kinds.values()))},
                        "gate_groups": len([g for g in groups if g]), "core_tiles": len(core_keys),
                        "lang_eng": "box:lang.eng is in the always-open core"},
               "gate_config": cfg.as_dict(), "gated_memory": mem, "arms": {k: list(v[:2]) + [v[2]] for k, v in plan.items()},
               "phases": [list(p) for p in PHASES], "sink_layers": a.sink_layers, "not_run": [], "states": {
                   "implemented": True, "qualified": False, "selected": False, "observed_used": False}}
    log(f"BAU: {mem['all_gated']['units']:,} gated units ({100 * mem['all_gated']['share_of_units']:.2f}%), "
        f"{mem['all_gated']['approx_bytes'] / 2**30:.2f} GiB approx / {mem['all_gated']['exact_bytes'] / 2**30:.2f} "
        f"GiB exact layout; {len(groups)} gate groups")
    summary_p = out / "summary.json"
    host = None
    try:
        lid, lrec = (lid_factory() if lid_factory else G.load_lid(a.lid_dir, verify=not a.skip_lid_sha))
        receipt["lid"] = lrec
        if a.dry_run:
            tok = TokJson(a.model)
            encode, decode = tok.encode, tok.decode
            receipt["tokenizer"] = "tokenizer.json (CPU, dry run)"
        else:
            from research.amoeba.carve import ops as O
            O.prepare_wiring(O.wired_arg(a.wired_gb))
            O.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb, every_s=1.0,
                                 min_avail_gb=a.min_avail_gb, hard_gb=a.watchdog_hard_gb,
                                 compressor_growth_gb=a.compressor_growth_gb, receipt=out / "watchdog-stop.json",
                                 watcher=True)
            O.setup_mlx(a.memory_gb, 2)
            if host_factory:
                host = host_factory(log)
            else:
                from research.amoeba.carve.host import Host
                host = Host(a.model, log=log, moe_units="neuron")
            if host.fingerprint != prep["fingerprint"]:
                raise SystemExit(f"partition of {prep['fingerprint']}, model is {host.fingerprint}")
            if a.assert_ple:
                tables = getattr(getattr(host, "adapter", None), "_tables", None)
                if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                        Path(a.model).expanduser().resolve() / "ple_rows.bin":
                    raise SystemExit("PLE offload is not active (FN rule: assert it)")
            receipt["wired"] = O.wire_weights(O.wired_arg(a.wired_gb), log, memory_gb=a.memory_gb)
            receipt["checkpoint"] = host.checkpoint
            receipt["watchdog"] = {"limit_gb": a.watchdog_gb, "swap_growth_gb": a.swap_growth_gb,
                                   "waiver_note": a.waiver_note or None, "min_avail_gb": a.min_avail_gb,
                                   "memory_gb": a.memory_gb}
            encode, decode = host.encode, host.decode
            receipt["tokenizer"] = "host"
        t_work = time.time()
        deadline = t_work + 60 * a.time_cap_min
        if a.stop_new_at:
            deadline = min(deadline, clock_deadline(a.stop_new_at, t_work))
            receipt["stop_new_at"] = {"clock": a.stop_new_at, "epoch": deadline,
                                      "local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(deadline))}
        sets = build_sets(meta, encode, a, log, prep)
        receipt["sets"] = {n: {"kind": s.kind, "docs": len(s.docs), "T": s.T, "spec": s.spec,
                               "prompt_tokens": sorted(set(s.prompts))[:3]} for n, s in sets.items()}
        receipt["held_path_substitutions"] = SUBSTITUTED
        nb = fit_religion(prep, meta, a, sets, log, purge_val=a.split == "held")
        receipt["religion_detector"] = nb.meta
        pol = G.GatePolicy(cfg, lid, nb)
        pol_nohin = G.GatePolicy(replace(cfg, religion_couples=()), lid, nb)
        rules = dict(G.RULES)
        prereg = {"rules": rules, "gate_config": cfg.as_dict(), "arms": receipt["arms"], "sets": receipt["sets"],
                  "religion_threshold": nb.threshold, "religion_threshold_decode": nb.threshold_decode,
                  "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "mlx2_source_revision": receipt["mlx2_source_revision"]}
        write_json(out / "prereg.json", prereg)
        receipt["prereg_sha256_12"] = sha12(out / "prereg.json")
        streams, glogs = streams_for(sets, {"auto": pol, "auto_nohin": pol_nohin}, couples, decode, focus, log)
        with open(out / "gate.jsonl", "w") as fh:
            for n, lgs in glogs.items():
                for i, lg in enumerate(lgs):
                    gm = G.gate_metrics(streams["auto"][n][i], streams["oracle"][n][i], sets[n].prompts[i])
                    fh.write(json.dumps({"set": n, "doc": i, "id": str(sets[n].rows[i].get("id")),
                                         "metrics": gm, **lg}, default=str) + "\n")
        grep, gpool = gate_report(sets, streams, glogs, tz.idx, groups, W, cfg, mem_args)
        receipt["gate"] = {"per_set": grep, "pool": gpool}
        log(f"gate: pooled missed-open {gpool['rate']}; " + ", ".join(
            f"{n} miss {r['missed_rate'] if r['missed_rate'] is None else round(r['missed_rate'], 4)} "
            f"gated {r['rows_fully_gated_share']:.2f}" for n, r in grep.items()))
        write_json(summary_p, receipt)
        if a.dry_run:
            receipt["not_run"] = ["every scored cell (dry run)"]
            receipt["status"] = "dry_run"
            return receipt
        # ---------------------------------------------------------------- scoring
        import amoeba_battery_stats as ST
        scorer = Scorer(host, tz, core_keys, groups, a.sink_layers, a.ce_batch, log)
        scorer_np = Scorer(host, tz, core_keys, groups_np, a.sink_layers, a.ce_batch, log)   # pair tiles closed
        (out / "ce").mkdir(exist_ok=True)
        receipt["cell_plan"] = [(arm, n) for ph in PHASES if ph != ("generations",) for n in sets for arm in ph
                                if plan[arm][2] is None or n in plan[arm][2]]
        docs_meta = {n: ST.doc_meta(s.rows, s.spec) for n, s in sets.items()}
        write_json(out / "perdoc.docs.json", docs_meta)
        ce = {}
        results = {}
        receipt["results"] = []                        # live: the finally-summary sees every finished cell
        gens = []
        receipt["generations"] = {"rows": 0, "correct": {}, "per_prompt": {}}
        rj = out / "results.jsonl"

        def compare(name):
            """auto-type arms vs the oracle on the same set, once both exist (paired per document)."""
            d = docs_meta[name]
            for arm in ("auto", "auto_nohin", "auto_nopairs"):
                r = results.get((arm, name))
                if r is None or "vs_oracle" in r or ("oracle", name) not in ce:
                    continue
                x, o = ce[(arm, name)], ce[("oracle", name)]
                r["vs_oracle"] = ST.paired([float(v.mean()) for v in x], [float(v.mean()) for v in o],
                                           d["groups"], d["strata"])
                with (out / "comparisons.jsonl").open("a") as fh:
                    fh.write(json.dumps({"arm": arm, "vs": "oracle", "set": name, **r["vs_oracle"]}) + "\n")

        def run(arm, name):
            src_, gv, only = plan[arm]
            s = sets[name]
            if only is not None and name not in only:
                return
            if time.time() > deadline:
                receipt["not_run"].append(f"{arm}:{name} (time cap)")
                return
            t0 = time.time()
            sc = scorer_np if gv == "nopairs" else scorer
            x = sc.ce(s.docs, None if src_ is None else streams[src_][name])
            np.save(out / "ce" / f"{arm}__{name}.npy", x)
            ce[(arm, name)] = x
            row = {"arm": arm, "set": name, "docs": len(s.docs), "T": s.T, "wall_s": round(time.time() - t0, 1),
                   "ce_mean": float(x.mean())}
            if arm != "full" and ("full", name) in ce:
                f = ce[("full", name)]
                d = docs_meta[name]
                row["dce"] = doc_stats(x, f, s.prompts, d["groups"], d["strata"])
                ST.append_perdoc(out / "perdoc.jsonl", name, arm, name,
                                 {"ce": x.astype(np.float64).sum(1), "clean_ce": f.astype(np.float64).sum(1),
                                  "n": np.full(len(x), x.shape[1])})
                if arm == "auto":
                    miss = []
                    for i, (a_, o_) in enumerate(zip(streams["auto"][name], streams["oracle"][name])):
                        # scored rows r = 1..T-2 (row r predicts target r + 1, CE index r - 1): k = r - 1
                        rows_ = np.flatnonzero((o_ & ~a_).any(1)[1:s.T - 1])
                        miss.extend((x[i, rows_] - f[i, rows_]).tolist())
                    row["missed_rows_dce"] = {"n": len(miss), "mean": float(np.mean(miss)) if miss else None}
            results[(arm, name)] = row
            receipt["results"].append(row)
            compare(name)
            checkpoint(receipt, out, a, t_start)
            with rj.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            dc = row.get("dce", {}).get("all", {})
            log(f"{arm} {name}: CE {row['ce_mean']:.4f}" + (f" dCE {dc['mean']:+.4f} +- {dc['se']:.4f}" if dc else "")
                + f" ({row['wall_s']}s)")
        import amoeba_nexttok_fn_window as NW
        for phase in PHASES:
            if phase == ("generations",):
                if time.time() > deadline:
                    receipt["not_run"].append("generations (time cap)")
                    continue
                sys.path.insert(0, str(ROOT / "scripts"))
                import amoeba_fn_box_trip as BT
                eos = BT.eos_ids(host)
                for pr in gen_prompts():
                    for arm in [x for x in a.gen_arms.split(",") if x]:
                        if time.time() > deadline:
                            receipt["not_run"].append(f"generation {arm}:{pr['id']} (time cap)")
                            continue
                        g = generate(host, scorer, pr, arm, pol, eos, a.gen_tokens, decode)
                        ok, top = output_correct(lid, pr, g["text"])
                        g.update(id=pr["id"], kind=pr["kind"], lang=pr["lang"], arm=arm, prompt=pr["text"],
                                 correct=ok, lid_top=top)
                        gens.append(g)
                        receipt["generations"] = gen_summary(gens)
                        checkpoint(receipt, out, a, t_start)
                        with (out / "generations.jsonl").open("a") as fh:
                            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
                        log(f"gen {arm} {pr['id']}: correct {ok} {top[:1]} opened {g['opened']} ({g['wall_s']}s)")
                NW.phase_end("generations", log, receipt)
                continue
            for name in sets:
                for arm in phase:
                    run(arm, name)
            NW.phase_end("phase " + "+".join(phase), log, receipt)
        # a window in which no cell was scored (deadline reached during the load) measured nothing: not a success
        receipt["status"] = "finished" if receipt["results"] else NO_CELLS
        return receipt
    except BaseException as e:                       # noqa: BLE001 - the summary is always written
        receipt["status"] = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        try:
            finish(receipt, out, a, t_start)
        finally:
            log_fh.close()


NO_CELLS = "no cell scored (deadline reached before the first cell)"


def clock_deadline(hhmm, now):
    """Epoch of the next local HH:MM after now; one more than 23 h away means today's has just passed (a window
    that started after the cutoff, e.g. 06:40 for 06:30): then now, so no cell starts. Anything earlier in the day
    (a daytime start at 10:00) gets the next occurrence, tomorrow's."""
    h, m = (int(x) for x in hhmm.split(":"))
    lt = time.localtime(now)
    t = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))
    t = t if t > now else t + 86400
    return now if t - now > 23 * 3600 else t


def gen_summary(gens):
    """Output-language correctness per arm (and per prompt kind); the decision compares only prompts both arms
    generated, so a time cap cannot favour one arm."""
    by = {}
    for g in gens:
        by.setdefault(g["id"], {})[g["arm"]] = g
    both = [v for v in by.values() if "full" in v and "auto" in v]
    out = {"rows": len(gens), "prompts_both_arms": len(both),
           "correct": {arm: sum(v[arm]["correct"] for v in both) for arm in ("full", "auto")} if both else {},
           "correct_by_kind": {}, "per_prompt": {f"{g['arm']}:{g['id']}": {"correct": g["correct"],
                                                                        "lid_top": g["lid_top"][:1],
                                                                        "opened": g["opened"]} for g in gens}}
    for v in both:
        k = v["full"]["kind"]
        d = out["correct_by_kind"].setdefault(k, {"full": 0, "auto": 0, "n": 0})
        d["n"] += 1
        d["full"] += int(v["full"]["correct"])
        d["auto"] += int(v["auto"]["correct"])
    return out


def checkpoint(receipt, out, a, t_start):
    """A provisional summary after every cell: a process the watchdog kills (os._exit / SIGKILL, no finally) still
    leaves the cells it finished and their verdict on disk."""
    try:
        finish(dict(receipt, status=f"running (provisional summary; {receipt.get('status')})"), out, a, t_start)
    except Exception as e:  # noqa: BLE001 - a checkpoint never stops the run
        print(f"checkpoint summary failed: {e}", flush=True)


def finish(receipt, out, a, t_start):
    """summary.json + summary.md from whatever ran (the verdict over the cells that exist)."""
    receipt["wall_s"] = round(time.time() - t_start, 1)
    plan = receipt.get("cell_plan") or []
    done = {(r["arm"], r["set"]) for r in receipt.get("results") or []}
    receipt["cells_not_run"] = [f"{a_}:{s}" for a_, s in plan if (a_, s) not in done]
    try:
        if receipt.get("dry_run"):
            raise ImportError("dry run: no MLX")
        from research.amoeba.carve import ops as O
        receipt.setdefault("watchdog", {}).update(
            {k: v for k, v in getattr(O, "WATCH", {}).items() if k.startswith(("peak_", "observed_"))})
        if not receipt.get("dry_run"):
            receipt["mlx_peak_gb"] = O.peak_gb()
    except Exception:  # noqa: BLE001
        pass
    sets = receipt.get("sets") or {}
    kinds = {n: v["kind"] for n, v in sets.items()}
    rows = {}
    for r in receipt.get("results") or []:
        if "dce" in r:
            rows[(r["arm"], r["set"])] = r["dce"]["all"]
        if "vs_oracle" in r and r["arm"] == "auto":
            rows[("auto-oracle", r["set"])] = r["vs_oracle"]
    gen = (receipt.get("generations") or {}).get("correct")
    gen = {"full": gen.get("full", 0), "auto": gen.get("auto", 0)} if gen and "auto" in gen and "full" in gen else None
    if kinds and not receipt.get("dry_run"):
        receipt["verdict"] = G.decide(rows, (receipt.get("gate") or {}).get("pool") or {}, gen, kinds,
                                      gate_sets=(receipt.get("gate") or {}).get("per_set"))
    else:
        receipt["verdict"] = {"verdict": "not scored", "label": G.RULES["label"]}
    receipt["focus_report"] = focus_report(receipt)
    write_json(out / "summary.json", receipt)
    (out / "summary.md").write_text(summary_md(receipt))


def focus_report(receipt):
    """arb / cmn / jpn / religion (and the focus sets) under every arm that scored them: dCE vs full."""
    rep = {}
    for r in receipt.get("results") or []:
        if r["set"] in ("arb", "cmn", "jpn", "hin", G.RELIGION, "mixed.switch") and "dce" in r:
            rep.setdefault(r["set"], {})[r["arm"]] = {"mean": r["dce"]["all"]["mean"], "se": r["dce"]["all"]["se"],
                                                     "ci95": r["dce"]["all"]["ci95"]}
    return rep


def summary_md(r):
    L = [f"# Language gate (BAU) on {r.get('checkpoint')}", "",
         f"mlx2 {r.get('mlx2_source_revision')}; partition {r.get('partition')} (partition.json "
         f"{r.get('partition_json_sha256_12')}); status {r.get('status')}; {r.get('wall_s')} s. "
         f"Label: {G.RULES['label']}.", ""]
    m = r.get("gated_memory") or {}
    if m:
        ag = m["all_gated"]
        L += [f"Gated: {ag['units']:,} units ({100 * ag.get('share_of_units', 0):.2f}%), "
              f"{ag['approx_bytes'] / 2**30:.2f} GiB approx, {ag['exact_bytes'] / 2**30:.2f} GiB exact layout.", ""]
    v = r.get("verdict") or {}
    L += [f"Verdict: **{v.get('verdict')}**" + (f" (failed: {', '.join(v.get('failed') or [])})" if v.get("failed") else ""), ""]
    if v.get("checks"):
        L += ["| set | kind | test | mean | 95% CI | bar | pass |", "|---|---|---|---|---|---|---|"]
        for c in v["checks"]:
            ci = c.get("ci95")
            mean = c.get("mean")
            L.append(f"| {c['set']} | {c['kind']} | {c['test']} | "
                     f"{'-' if mean is None else (f'{mean:+.4f}' if isinstance(mean, float) else mean)} | "
                     f"{'-' if not ci else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} | {c['bar']} | "
                     f"{'yes' if c['pass'] else 'NO'} |")
    g = (r.get("gate") or {}).get("per_set") or {}
    if g:
        L += ["", "| set | prefill exact | missed rate | rows fully gated | never-opened GiB (exact) | fail-open reqs |",
              "|---|---|---|---|---|---|"]
        for n, x in g.items():
            mr = x["missed_rate"]
            L.append(f"| {n} | {x['prefill_exact_rate']:.2f} | {'-' if mr is None else f'{mr:.4f}'} | "
                     f"{x['rows_fully_gated_share']:.2f} | {x['never_opened_bytes_mean']['exact_bytes'] / 2**30:.2f} | "
                     f"{x['fail_open_requests']} |")
    fr = r.get("focus_report") or {}
    if fr:
        L += ["", "Focus sets (dCE vs full, all targets):", ""]
        for s, arms in fr.items():
            L.append(f"- {s}: " + ", ".join(f"{k} {x['mean']:+.4f}" for k, x in sorted(arms.items())))
    if r.get("not_run"):
        L += ["", "Not run: " + "; ".join(r["not_run"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(3 if main().get("status") == NO_CELLS else 0)
