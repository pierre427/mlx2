"""H60: a small memory LM trained only on recall-shaped QA synthesized from a corpus (MeMo, arXiv 2605.15156; the fair
retry of H51 / H56, whose tiles trained on raw text gained trained-document NLL but no question recall). A frozen core
asks the memory questions in text and reads its answer (the bus arm comes later: see MemoryEndpoint).

Three resumable subcommands, each with a receipt (`<name>@<fingerprint>` checkpoints, source_revision, host, mlx):

  synth   For every TRAINED document only (held documents never enter synthesis; asserted from split.json), a local
          generator model (default mlx-community/Qwen3-4B-4bit on the M3) writes recall-shaped QA from windows of the
          document: direct facts, indirect (inferred) facts, both directions of entity relations (against the reversal
          curse), all self-contained (the entity named, no pronouns, no "the excerpt"); then a few cross-document
          syntheses over pairs of trained documents that share a rare named entity. Filters: answer non-empty and
          short, answer grounded in the source text, question names its entity, no deictic references, dedupe.
          LEAKAGE: an item is dropped (and counted) when its question is near-identical to any eval question or
          rephrase (recall.jsonl, crossdoc.jsonl): all-token Jaccard >= --leak-threshold (0.6), or content-token
          Jaccard >= --leak-fact-threshold (0.5) with the same answer; the overlap histogram is in the receipt.
          Budget: --items-per-doc (cap), --items-per-1k (scale with length), --min-items. Resumable per document.
          Generator backends: mlx_lm (load + batch_generate / generate) when importable, else `ens`: this module's
          KV-cached, left-padded batched greedy decoder over research/ens/qwen3.py blocks with the checkpoint's
          affine quantization (tested on CPU against the uncached reference forward).
  train   LoRA (rank 64, all 7 projections, every layer) on Qwen3-0.6B, the synthesized QA only: chat format with an
          empty think block, the eval prompt (corpus_tile_eval.prompt_ids), loss on the answer tokens + <|im_end|>.
          AdamW lr 2e-4 (training lessons 4: LoRA SFT 1e-4..2e-4), clip 1.0, 5% warmup, cosine to 0.1 x lr; 5% of the
          QA rows held out for validation; every --track-every steps: train loss, held-QA loss, and teacher-forced
          answer NLL of the eval recall questions on trained vs held documents (gain over base, trained minus held:
          the H56-style check that catches memorising-without-recall). Best checkpoint by held-QA loss. Resumable.
  eval    The pinned eval questions (question and rephrase fields) through three arms: base Qwen3-0.6B closed book;
          the memory model; BM25 (pure Python) over the TRAINED corpus text with the top-k passages pasted as context
          for base Qwen3-0.6B, at fixed context budgets (--bm25 KxP: k passages of P tokens). Cells as qa_eval
          (recall.{train,held_out}.{field}, synthesis.{mix}.{field}, graph_mc.{mix}.{field}), each with lenient
          (corpus_tile_eval.correct) and strict (readout_check.Strict over recall rows) scores. Base arm decoded by
          corpus_tile_eval.generate (as H56), memory and BM25 arms by the cached decoder (recorded per arm). Kill (in
          the receipt, with document-level paired bootstrap CIs and MDE): memory recall on trained documents no better
          than BM25 + base at the matched budget (--kill-bm25), or no trained-minus-held gap over base (difference in
          differences: it answers from priors); a test kills only when its CI excludes a positive effect, and a
          missing arm or split is "undecided", never "not killed". Also reported on covered questions only.
          Synth gates (exit 4): fewer than 20 kept items per 100 trained documents, or kept/raw < 0.2, checked after
          25 documents and at the end; train refuses an empty qa.jsonl or zero steps (exit 4).

Everything here is private-corpus evidence: receipts carry `label: private_corpus: never publish`; logs name documents
only by sha256[:12]; nothing is sent to any network service (HF_HUB_OFFLINE in the runner).

  python -m research.ens.qa_memory synth --corpus <dir> --questions <eval dir> --out <dir>
  python -m research.ens.qa_memory train --synth <dir> --corpus <dir> --questions <eval dir> --out <dir>
  python -m research.ens.qa_memory eval --run <train dir> --questions <eval dir>

Known caveats (review 2026-11-26): research.ens.m3 stops a job by SIGKILL of its process group, so the SIGTERM
handlers here never run; a hard stop loses up to one checkpoint interval of training, and best.safetensors may be
newer than the restored log (training is deterministic, so a resume largely re-derives it). load_quantized reads
group size and bits but ignores quantization.mode (fine for the affine Qwen3-4B-4bit; check before another format).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import random
import re
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import corpus_tile as CT
from research.ens import corpus_tile_eval as E
from research.ens import lm_bus as LB
from research.ens import physics_bus as PB
from research.ens import qwen3 as Q

LABEL = "private_corpus: never publish"
RECIPE = "ens.h60.qa_memory.v1"
GENERATOR = "mlx-community/Qwen3-4B-4bit"
TYPES = ("direct", "indirect", "reverse", "crossdoc")
STOP = frozenset("""a an the of in on at to for from by with and or but is are was were be been being as that this these
those which who whom whose what when where why how does do did has have had it its into than then there their they them
he she his her him not no yes can could would should will may might about over under between also such any each per
via using used use one two""".split())
DEICTIC = re.compile(r"\b(the|this|that|these)\s+(excerpt|document|text|passage|article|lecture|speaker|author|"
                     r"transcript|paper|section|entry|note|record|source|video|talk|course|context)s?\b", re.I)
PRONOUN_ANY = re.compile(r"\b(he|she|him|her|hers|himself|herself)\b", re.I)
PRONOUN_START = re.compile(r"^\s*(?:\w+\s+){0,2}(they|them|their|it|its)\b", re.I)
NON_ANSWERS = {"", "unknown", "not stated", "n a", "none", "not mentioned", "not specified", "unclear"}


def hid(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def sha16(p: Path) -> str:
    return CT.sha16(p)


def say(*a):
    print(*a, flush=True)


_SPECIAL_TEXT = re.compile(r"<(\|?)(im_start|im_end|endoftext|/?think)")


def text_ids(tok, s):
    """Untrusted text (corpus documents, synthesized QA) with special-token parsing OFF, so literal control-token text
    in a document (e.g. "<|im_end|>") stays text. Tokenizers without the switch (test tokenizers) get the control
    strings defanged with a zero-width space instead."""
    t = getattr(tok, "t", None)
    if t is not None and hasattr(t, "encode_special_tokens"):
        prev = t.encode_special_tokens
        t.encode_special_tokens = True
        try:
            return t.encode(s or "", add_special_tokens=False).ids
        finally:
            t.encode_special_tokens = prev
    return tok.encode(_SPECIAL_TEXT.sub(lambda m: "<\u200b" + m[1] + m[2], s or ""))


def chat_ids(tok, user_text):
    """The physics_qa.chat frame (thinking off) with the user turn encoded as untrusted text."""
    return (tok.encode("<|im_start|>user\n") + text_ids(tok, user_text) +
            tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))


# ================================================================== corpus, split guard
def load_inputs(corpus: Path):
    """(docs by id, train ids, held ids, hashes). Refuses an overlapping split and unknown ids."""
    corpus = Path(corpus).expanduser()
    docs = [json.loads(l) for l in open(corpus / "corpus.jsonl")]
    split = json.loads((corpus / "split.json").read_text())
    train_ids, held_ids = set(split["train"]), set(split["held_out"])
    if train_ids & held_ids:
        raise SystemExit("split.json: a document is both train and held out")
    by_id = {d["id"]: d for d in docs}
    if not train_ids <= set(by_id):
        raise SystemExit("split.json names trained documents missing from corpus.jsonl")
    hashes = {n: sha16(corpus / n) for n in ("corpus.jsonl", "split.json", "graph.json") if (corpus / n).exists()}
    return by_id, train_ids, held_ids, hashes


def assert_trained_only(doc_ids, train_ids, held_ids):
    """The split guard: synthesis may touch trained documents only."""
    for d in doc_ids:
        if d in held_ids or d not in train_ids:
            raise AssertionError(f"held or unknown document {hid(d)} reached QA synthesis")


def load_questions(qdir: Path):
    qdir = Path(qdir).expanduser()
    rec = [json.loads(l) for l in open(qdir / "recall.jsonl")] if (qdir / "recall.jsonl").exists() else []
    xd = [json.loads(l) for l in open(qdir / "crossdoc.jsonl")] if (qdir / "crossdoc.jsonl").exists() else []
    if not rec and not xd:
        raise FileNotFoundError(f"{qdir}: missing recall.jsonl and crossdoc.jsonl")
    return rec, xd


def check_doc_split(rec, train_ids, held_ids):
    """Every recall question's doc_split must agree with split.json (as readout_check.doc_split); refuse otherwise."""
    for r in rec:
        want = "train" if r.get("doc_id") in train_ids else "held_out" if r.get("doc_id") in held_ids else None
        if want is None or r.get("doc_split") != want:
            raise SystemExit(f"question {hid(r.get('qid'))}: doc_split disagrees with split.json")


def questions_sha12(qdir: Path) -> str:
    """qa_eval's question-set hash (score2 cells), so cells compare with H56 receipts."""
    d = hashlib.sha256()
    for f in (Path(qdir) / "recall.jsonl", Path(qdir) / "crossdoc.jsonl"):
        if f.is_file():
            d.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    return d.hexdigest()[:12]


# ================================================================== text utilities
def content_tokens(text) -> set:
    return {t for t in E._norm(text).split() if t not in STOP and (len(t) > 1 or t.isdigit())}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def names_entity(q, title="", window=""):
    """The question names its entity: a capitalised word after the first, a number, an acronym or quoted term, or a
    content word of the document title."""
    words = re.findall(r"[A-Za-z0-9][\w\-\.']*", q)
    for i, w in enumerate(words):
        if any(ch.isdigit() for ch in w) or (len(w) > 1 and w.isupper()):
            return True
        if i > 0 and w[0].isupper() and w not in ("I",):
            return True
    if re.search(r"[\"“'‘][^\"”'’]{2,}[\"”'’]", q):
        return True
    tw = {t for t in content_tokens(title) if len(t) >= 4}
    return bool(tw & content_tokens(q))


def grounded(answer, source, min_share=0.5):
    """Share of the answer's content tokens present in the source text >= min_share (anti-hallucination)."""
    a = content_tokens(answer)
    if not a:
        return E._norm(answer) in E._norm(source)
    s = set(E._norm(source).split())
    return sum(t in s for t in a) / len(a) >= min_share


def check_item(it, title, source, max_answer_words=10, max_answer_chars=80):
    """-> None if the item passes, else the drop reason."""
    q, a = (it.get("q") or "").strip(), (it.get("a") or "").strip()
    an = E._norm(a)
    if an in NON_ANSWERS:
        return "empty_answer"
    if len(a.split()) > max_answer_words or len(a) > max_answer_chars:
        return "long_answer"
    nq = len(q.split())
    if nq < 4 or nq > 45 or not q.endswith("?"):
        return "bad_question"
    if DEICTIC.search(q) or PRONOUN_ANY.search(q) or PRONOUN_START.search(q):
        return "deictic"
    if not names_entity(q, title, source):
        return "no_entity"
    if E._has(E._norm(q), a):
        return "answer_in_question"
    if not grounded(a, source):
        return "ungrounded"
    return None


def parse_items(text, n_max=None):
    """JSON-lines items {"type","q","a"} from generator output (tolerates fences, trailing commas, a JSON list)."""
    out = []
    body = re.sub(r"```(?:json)?", "", text or "")
    cands = []
    try:
        v = json.loads(body.strip())
        cands = v if isinstance(v, list) else [v]
    except ValueError:
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                cands.append(json.loads(line))
            except ValueError:
                m = re.search(r'"q(?:uestion)?"\s*:\s*"(.+?)"\s*,\s*"a(?:nswer)?"\s*:\s*"(.+?)"', line)
                if m:
                    cands.append({"q": m[1], "a": m[2]})
    for c in cands:
        if not isinstance(c, dict):
            continue
        q, a = c.get("q", c.get("question")), c.get("a", c.get("answer"))
        if not isinstance(q, str) or not isinstance(a, (str, int, float)):
            continue
        t = str(c.get("type", "direct")).lower().strip()
        out.append({"type": t if t in TYPES else "direct", "q": q.strip(), "a": str(a).strip()})
    return out[:n_max] if n_max else out


def dedupe(items, near=0.85):
    """Exact (normalised question) and near duplicates (content Jaccard >= near with the same normalised answer)
    within a document. Keeps the first. -> (kept, n_dropped)."""
    kept, seen, by_doc = [], set(), collections.defaultdict(list)
    drop = 0
    for it in items:
        key = (it["doc_id"], E._norm(it["q"]))
        if key in seen:
            drop += 1
            continue
        toks, an = content_tokens(it["q"]), E._norm(it["a"])
        if any(an == a2 and jaccard(toks, t2) >= near for t2, a2 in by_doc[it["doc_id"]]):
            drop += 1
            continue
        seen.add(key)
        by_doc[it["doc_id"]].append((toks, an))
        kept.append(it)
    return kept, drop


LEAK_BINS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def eval_question_texts(rec, xd):
    """[(text, answer, aliases)] for every eval question and rephrase (recall + crossdoc)."""
    return [(r[f], r.get("answer") or r.get("choice_answer") or "", list(r.get("aliases") or []))
            for r in rec + xd for f in ("question", "rephrase") if r.get(f)]


def all_tokens(text) -> set:
    return set(E._norm(text).split())


def leakage_filter(items, eval_items, threshold=0.6, fact_threshold=0.5):
    """Drop an item when its question is near-identical to an eval question or rephrase:
      wording  Jaccard of all normalised tokens >= threshold (whatever the answer), or
      fact     Jaccard of content tokens >= fact_threshold AND the item's answer scores correct against that eval
               question's answer (the same fact asked in nearly the same words).
    eval_items: [(text, answer, aliases)] (plain strings are accepted, with no answer).
    -> (kept, report {n_dropped, n_dropped_wording, n_dropped_fact, thresholds, overlap histogram})."""
    ev = [(x, "", []) if isinstance(x, str) else x for x in eval_items]
    ev_all = [all_tokens(t) for t, _, _ in ev]
    ev_con = [content_tokens(t) for t, _, _ in ev]
    index = collections.defaultdict(set)
    for i, s in enumerate(ev_all):
        for t in s:
            index[t].add(i)
    kept, hist, n_word, n_fact = [], collections.Counter(), 0, 0
    for it in items:
        sa, sc = all_tokens(it["q"]), content_tokens(it["q"])
        cand = set().union(*(index.get(t, set()) for t in sa)) if sa else set()
        m = max((jaccard(sa, ev_all[i]) for i in cand), default=0.0)
        for b in LEAK_BINS:
            if m >= b:
                hist[f">={b}"] += 1
        if m >= threshold:
            n_word += 1
            continue
        if any(ev[i][1] and jaccard(sc, ev_con[i]) >= fact_threshold and E.correct(it["a"], ev[i][1], ev[i][2])
               for i in cand):
            n_fact += 1
            continue
        kept.append(it | {"leak_overlap": round(m, 3)})
    return kept, {"threshold_wording": threshold, "threshold_fact": fact_threshold, "n_checked": len(items),
                  "n_dropped": n_word + n_fact, "n_dropped_wording": n_word, "n_dropped_fact": n_fact,
                  "n_eval_texts": len(ev), "max_wording_overlap_at_least": {f">={b}": hist[f">={b}"] for b in LEAK_BINS}}


# ================================================================== window plan, prompts
def item_budget(n_tokens, items_per_doc=48, items_per_1k=4.0, min_items=6):
    return int(max(min_items, min(items_per_doc, round(items_per_1k * n_tokens / 1000))))


def plan_windows(n_windows, budget, min_per_call=3, max_per_call=8):
    """[(window index, n items)]: as many evenly spaced windows as the budget covers at >= min_per_call items each,
    the budget spread over them (<= max_per_call each)."""
    if n_windows <= 0 or budget <= 0:
        return []
    n_use = max(1, min(n_windows, budget // min_per_call))
    per = min(max_per_call, -(-budget // n_use))
    idx = sorted({int(round(i * (n_windows - 1) / max(1, n_use - 1))) for i in range(n_use)}) if n_use > 1 else [0]
    out, left = [], budget
    for i in idx:
        k = min(per, left)
        if k <= 0:
            break
        out.append((i, k))
        left -= k
    return out


DOC_PROMPT = """Below is an excerpt from a document titled "{title}".
Write {n} question-answer pairs that test memory of the facts in this excerpt.
Rules:
- Every question must be self-contained: name the person, method, model, paper, project or setting explicitly. Never use pronouns or phrases like "the excerpt", "the document", "the speaker", "the author" or "this lecture".
- Answers are short: a name, number, date, term or phrase of at most 8 words, taken from the excerpt.
- Mix the types: "direct" (a fact stated in the excerpt), "indirect" (a fact that follows from combining two statements in the excerpt), and "reverse" (when a fact links two named things, ask it in both directions as two items: if one asks "Who proposed X?" with answer Y, another asks "What did Y propose?" with answer X).
Output exactly one JSON object per line and nothing else, in the form {{"type": "direct", "q": "...", "a": "..."}}

Excerpt:
{text}"""

CROSS_PROMPT = """Below are excerpts from two documents, "{t1}" and "{t2}". Both mention {entity}.
Write {n} question-answer pairs whose answers need facts from BOTH excerpts (for example, how the two documents relate to {entity}, or a comparison).
Rules: every question names {entity} and the other things it asks about explicitly; no pronouns; no phrases like "the excerpt" or "the document". Answers are short (at most 8 words).
Output exactly one JSON object per line and nothing else, in the form {{"type": "crossdoc", "q": "...", "a": "..."}}

Excerpt 1:
{x1}

Excerpt 2:
{x2}"""


def chat_prompt(text):
    """Qwen3 chat, thinking off (the same frame as physics_qa.chat)."""
    from research.ens import physics_qa as QA
    return QA.chat(text)


def doc_windows(tok, doc, window_tokens):
    ids = text_ids(tok, doc.get("text") or "")
    return [ids[j:j + window_tokens] for j in range(0, max(1, len(ids)), window_tokens)] if ids else [], len(ids)


_ENT = re.compile(r"\b(?:[A-Z][a-zA-Z0-9\-]+(?:\s+(?:of|for|de)?\s*[A-Z][a-zA-Z0-9\-]+){1,3}|[A-Z]{2,}[0-9]*"
                  r"(?:-[A-Za-z0-9]+)?)\b")


def doc_entities(text, min_count=2):
    c = collections.Counter(m.group(0).strip() for m in _ENT.finditer(text or ""))
    return {e for e, n in c.items() if n >= min_count and len(e) >= 3 and e.lower() not in STOP}


def crossdoc_pairs(docs, train_ids, n_pairs, seed=0, max_df=8, per_doc=3):
    """[(doc a, doc b, entity)]: trained documents sharing a rare named entity (in 2..max_df trained documents)."""
    ids = sorted(train_ids)
    assert_trained_only(ids, train_ids, set())
    ents = {d: doc_entities(docs[d].get("text")) for d in ids}
    df = collections.defaultdict(list)
    for d in ids:
        for e in ents[d]:
            df[e].append(d)
    cands = sorted((e, ds) for e, ds in df.items() if 2 <= len(ds) <= max_df)
    rng = random.Random(seed)
    rng.shuffle(cands)
    used, seen, out = collections.Counter(), set(), []
    for e, ds in cands:
        if len(out) >= n_pairs:
            break
        ds = list(ds)
        rng.shuffle(ds)
        for a, b in ((ds[i], ds[j]) for i in range(len(ds)) for j in range(i + 1, len(ds))):
            key = tuple(sorted((a, b)))
            if key in seen or used[a] >= per_doc or used[b] >= per_doc:
                continue
            seen.add(key)
            used[a] += 1
            used[b] += 1
            out.append((key[0], key[1], e))
            break
    return out


def excerpt_around(tok, text, entity, n_tok=600):
    i = max(0, (text or "").find(entity))
    s = text[max(0, i - 2000): i + 2000]
    ids = text_ids(tok, s)
    j = len(text_ids(tok, text[max(0, i - 2000): i])) if i else 0
    lo = max(0, min(j - n_tok // 2, len(ids) - n_tok))
    return tok.t.decode(ids[lo:lo + n_tok])


# ================================================================== generation (ens backend: KV-cached batched greedy)
def load_quantized(path: Path):
    """A research/ens/qwen3.py Qwen3 from an HF/MLX snapshot, with the checkpoint's affine quantization (mlx-community
    config `quantization`, per-module overrides honoured; a module is quantized only when its .scales exist)."""
    path = Path(path)
    cfgd = json.loads((path / "config.json").read_text())
    c = Q.Qwen3Config.from_hf(cfgd)
    m = Q.Qwen3(c)
    files = sorted(path.glob("model*.safetensors"))
    w = {}
    for f in files:
        w.update(mx.load(str(f)))
    if c.tie_word_embeddings:
        w = {k: v for k, v in w.items() if not k.startswith("lm_head.")}
    qc = cfgd.get("quantization") or cfgd.get("quantization_config")
    if qc:
        def pred(p, _mod):
            if f"{p}.scales" not in w:
                return False
            ov = qc.get(p)
            return ov if isinstance(ov, dict) else True
        nn.quantize(m, group_size=qc["group_size"], bits=qc["bits"], class_predicate=pred)
    m.load_weights(list(w.items()))
    m.freeze()
    mx.eval(m.parameters())
    name = path.parent.parent.name.split("--")[-1] if path.parent.name == "snapshots" else path.name
    return m, f"{name}@{Q._fingerprint(path)}"


def _proj(mod, name, x, lora):
    y = getattr(mod, name)(x)
    return y + lora[name](x) if lora and name in lora else y


def cached_generate(model, tok, prompts, max_new=40, lora=None, chunk=512):
    """Greedy decoding with a preallocated KV cache, prompts left-padded (pad keys masked; RoPE is relative, so the
    shift changes nothing). lora: None or {f"b{i}": {proj: callable}} (MemoryAdapter.lora). Output texts as
    corpus_tile_eval.generate (the stop token included in the decoded ids)."""
    B = len(prompts)
    c = model.config
    Lp = max(len(p) for p in prompts)
    S = Lp + max_new
    pad = tok.pad
    rows = [[pad] * (Lp - len(p)) + list(p) for p in prompts]
    valid = mx.array([[0] * (Lp - len(p)) + [1] * len(p) + [1] * max_new for p in prompts]) > 0
    ends = {tok.t.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>")} - {None}
    K, hd, H = c.num_key_value_heads, c.head_dim, c.num_attention_heads
    caches = None

    def step(toks, off):
        nonlocal caches
        h = model.model.embed_tokens(toks)
        t = toks.shape[1]
        if caches is None:
            caches = [[mx.zeros((B, K, S, hd), dtype=h.dtype), mx.zeros((B, K, S, hd), dtype=h.dtype)]
                      for _ in model.model.layers]
        qpos = off + mx.arange(t)[:, None]
        kpos = mx.arange(off + t)[None, :]
        allowed = (kpos <= qpos)[None, None] & valid[:, None, None, :off + t]
        # finite, dtype-safe: a pad query whose keys are all masked gets a uniform (finite) row, never NaN
        mask = mx.where(allowed, 0.0, 0.5 * float(mx.finfo(h.dtype).min)).astype(h.dtype)
        for i, blk in enumerate(model.model.layers):
            lo = lora.get(f"b{i}") if lora else None
            att = blk.self_attn
            x = blk.input_layernorm(h)
            q = att.q_norm(_proj(att, "q_proj", x, lo).reshape(B, t, H, hd)).transpose(0, 2, 1, 3)
            k = att.k_norm(_proj(att, "k_proj", x, lo).reshape(B, t, K, hd)).transpose(0, 2, 1, 3)
            v = _proj(att, "v_proj", x, lo).reshape(B, t, K, hd).transpose(0, 2, 1, 3)
            q, k = att.rope(q, offset=off), att.rope(k, offset=off)
            caches[i][0][:, :, off:off + t, :] = k
            caches[i][1][:, :, off:off + t, :] = v
            a = mx.fast.scaled_dot_product_attention(q, caches[i][0][:, :, :off + t], caches[i][1][:, :, :off + t],
                                                     scale=hd ** -0.5, mask=mask)
            h = h + _proj(att, "o_proj", a.transpose(0, 2, 1, 3).reshape(B, t, -1), lo)
            h = h + blk.mlp(blk.post_attention_layernorm(h), lo)
        return model.head(h[:, -1:, :])[:, 0]

    T = mx.array(rows)
    logits = None
    for s in range(0, Lp, chunk):
        logits = step(T[:, s:s + chunk], s)
        mx.eval(logits, caches)
    out = [[] for _ in range(B)]
    done = [False] * B
    for n in range(max_new):
        nxt = mx.argmax(logits, axis=-1)
        ids = nxt.tolist()
        for b in range(B):
            if not done[b]:
                out[b].append(ids[b])
                done[b] = ids[b] in ends
        if all(done) or n == max_new - 1:
            break
        logits = step(nxt[:, None], Lp + n)
        mx.eval(logits)
    return [tok.t.decode(o) for o in out]


class Generator:
    """Text generator over chat prompts (token ids). backend: auto (mlx_lm if importable, else ens), mlx_lm, ens."""

    def __init__(self, repo, backend="auto"):
        self.path = Q.snapshot_dir(repo)
        self.tok = LB.Tok(self.path)
        if backend == "auto":
            try:
                import mlx_lm  # noqa: F401
                backend = "mlx_lm"
            except ImportError:
                backend = "ens"
        self.backend = backend
        name = self.path.parent.parent.name.split("--")[-1]
        if backend == "mlx_lm":
            import mlx_lm
            self.model, self._tw = mlx_lm.load(str(self.path))
            self.checkpoint = f"{name}@{Q._fingerprint(self.path)}"
            self.mlx_lm_version = getattr(mlx_lm, "__version__", None)
        else:
            self.model, self.checkpoint = load_quantized(self.path)
            self.mlx_lm_version = None
        self._batched = None

    def __call__(self, prompts, max_new):
        if self.backend == "ens":
            return cached_generate(self.model, self.tok, prompts, max_new=max_new)
        import mlx_lm
        if self._batched is not False:
            try:
                from mlx_lm import batch_generate
                r = batch_generate(self.model, self._tw, prompts, max_tokens=max_new, verbose=False)
                self._batched = True
                return list(r.texts)
            except (ImportError, TypeError, AttributeError) as e:      # older mlx_lm: one prompt at a time
                say(f"mlx_lm batch_generate unavailable ({type(e).__name__}); single-stream generation")
                self._batched = False
        return [mlx_lm.generate(self.model, self._tw, prompt=p, max_tokens=max_new, verbose=False) for p in prompts]


# ================================================================== synth
_STOP = {"requested": False}


def _on_term(_s, _f):
    _STOP["requested"] = True


def _committed(path: Path, key: str) -> dict:
    """Rows of an append-only jsonl keyed by `key`; a partial last line (interrupted write) is dropped."""
    out = {}
    if path.exists():
        good = []
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            try:
                r = json.loads(line)
            except ValueError:
                break
            out[r[key]] = r
            good.append(line)
        path.write_bytes(b"".join(good))
    return out


def synth_tasks(docs, train_ids, held_ids, tok, a):
    """[(doc_id, window index, n items, window text)] for every trained document in a fixed order."""
    ids = sorted(train_ids)
    if a.limit_docs:
        ids = ids[:a.limit_docs]
    assert_trained_only(ids, train_ids, held_ids)
    tasks = []
    for d in ids:
        wins, n_tok = doc_windows(tok, docs[d], a.window_tokens)
        budget = item_budget(n_tok, a.items_per_doc, a.items_per_1k, a.min_items)
        for wi, k in plan_windows(len(wins), budget, a.min_per_call, a.max_per_call):
            tasks.append((d, wi, k, tok.t.decode(wins[wi])))
    return tasks


def quality_gate(n_final, n_raw, n_docs, min_per_100=20.0, min_ratio=0.2):
    """None if the kept QA is usable, else the reason: fewer than min_per_100 kept items per 100 trained documents
    processed, or kept / raw below min_ratio (an empty or mostly filtered QA set would train nothing and read as a
    kill)."""
    if n_docs <= 0:
        return "no trained documents processed"
    if n_final * 100 < min_per_100 * n_docs:
        return f"{n_final} kept items for {n_docs} documents (< {min_per_100:g} per 100 documents)"
    if n_raw <= 0 or n_final / n_raw < min_ratio:
        return f"kept/raw {n_final}/{n_raw} < {min_ratio:g}"
    return None


def _gate_fail(out, why, counts, n_docs, where):
    C.write_json(out / "synth_gate_failed.json", {"label": LABEL, "where": where, "reason": why, "n_documents": n_docs,
                                                  **{k: counts[k] for k in ("n_raw_items", "n_final",
                                                                            "dropped_by_filter", "dropped_duplicates")},
                                                  "leakage": counts["leakage"]})
    say(f"synth: QA quality gate failed ({where}): {why}; stopping")
    return 4


def run_synth(a, gen, tok, docs, train_ids, held_ids, eval_texts, hashes, out: Path, gen_info: dict, rec=()) -> int:
    """The synth loop with an injected generator gen(prompt id lists, max_new) -> texts (tests pass a fake)."""
    raw_p, xraw_p = out / "raw_docs.jsonl", out / "raw_crossdoc.jsonl"
    done = _committed(raw_p, "doc_id")
    xdone = _committed(xraw_p, "pair")
    tasks = synth_tasks(docs, train_ids, held_ids, tok, a)
    by_doc = collections.defaultdict(list)
    for t in tasks:
        by_doc[t[0]].append(t)
    todo = [t for t in tasks if t[0] not in done]
    say(f"synth: {len(by_doc)} trained documents, {len(tasks)} windows, {len(done)} documents already done")
    t0, n_batches, n_items_seen = time.time(), 0, sum(len(r["items"]) for r in done.values())
    pending = collections.defaultdict(list)
    gated = False
    _STOP["requested"] = False
    old = signal.signal(signal.SIGTERM, _on_term)
    try:
        with raw_p.open("a") as fraw:
            for i in range(0, len(todo), a.gen_batch):
                if _STOP["requested"] or (a.max_seconds and time.time() - t0 > a.max_seconds):
                    say("synth: stopped (resumable)")
                    return 3
                sub = todo[i:i + a.gen_batch]
                assert_trained_only([d for d, *_ in sub], train_ids, held_ids)
                prompts = [chat_ids(tok, DOC_PROMPT.format(title=docs[d].get("title") or "", n=k, text=txt))
                           for d, _wi, k, txt in sub]
                texts = gen(prompts, min(a.max_new, 64 * max(k for _, _, k, _ in sub) + 48))
                n_batches += 1
                for (d, wi, k, txt), o in zip(sub, texts):
                    items = parse_items(o, n_max=k + 2)
                    n_items_seen += len(items)
                    pending[d].append({"window": wi, "n_asked": k, "items": items, "n_out_tok": len(tok.encode(o))})
                for d in list(pending):
                    if len(pending[d]) == len(by_doc[d]):
                        w = sorted(pending.pop(d), key=lambda r: r["window"])
                        row = {"doc_id": d, "windows": [r["window"] for r in w],
                               "items": [it | {"window": r["window"]} for r in w for it in r["items"]],
                               "n_asked": sum(r["n_asked"] for r in w), "n_out_tok": sum(r["n_out_tok"] for r in w)}
                        fraw.write(json.dumps(row) + "\n")
                        fraw.flush()
                        done[d] = row
                if not gated and len(done) >= a.gate_docs:
                    gated = True
                    _, cnt = filter_all(a, tok, docs, train_ids, held_ids, done, {}, [], eval_texts)
                    why = quality_gate(cnt["n_final"], cnt["n_raw_items"], len(done), a.min_items_per_100,
                                       a.min_kept_ratio)
                    if why:
                        return _gate_fail(out, why, cnt, len(done), f"after {len(done)} documents")
                    say(f"synth: quality gate ok after {len(done)} documents ({cnt['n_final']}/{cnt['n_raw_items']} kept)")
                if n_batches == a.sanity_batches and n_items_seen == 0:
                    say(f"synth: generator produced no parseable items in {n_batches} batches; stopping")
                    return 4
                if n_batches % 10 == 0:
                    say(f"synth: {len(done)}/{len(by_doc)} documents, {n_items_seen} raw items, "
                        f"{time.time() - t0:.0f} s")
        # cross-document syntheses over pairs of trained documents sharing an entity
        pairs = crossdoc_pairs(docs, train_ids, a.crossdoc_pairs, seed=a.seed)
        xtodo = [p for p in pairs if "|".join(p[:2]) not in xdone]
        with xraw_p.open("a") as fx:
            for i in range(0, len(xtodo), a.gen_batch):
                if _STOP["requested"] or (a.max_seconds and time.time() - t0 > a.max_seconds):
                    say("synth: stopped in the cross-document stage (resumable)")
                    return 3
                sub = xtodo[i:i + a.gen_batch]
                assert_trained_only([x for p in sub for x in p[:2]], train_ids, held_ids)
                srcs = [(excerpt_around(tok, docs[d1]["text"], e, a.cross_tokens),
                         excerpt_around(tok, docs[d2]["text"], e, a.cross_tokens)) for d1, d2, e in sub]
                prompts = [chat_ids(tok, CROSS_PROMPT.format(
                    t1=docs[d1].get("title") or "", t2=docs[d2].get("title") or "", entity=e, n=a.crossdoc_items,
                    x1=x1, x2=x2)) for (d1, d2, e), (x1, x2) in zip(sub, srcs)]
                texts = gen(prompts, min(a.max_new, 64 * a.crossdoc_items + 48))
                for (d1, d2, e), o in zip(sub, texts):
                    row = {"pair": f"{d1}|{d2}", "doc_ids": [d1, d2], "entity_sha12": hid(e),
                           "items": [it | {"type": "crossdoc"} for it in parse_items(o, n_max=a.crossdoc_items + 1)]}
                    fx.write(json.dumps(row) + "\n")
                    fx.flush()
                    xdone[row["pair"]] = row
    finally:
        signal.signal(signal.SIGTERM, old)
    rep = finalize(a, tok, docs, train_ids, held_ids, done, xdone, pairs, eval_texts, out, rec)
    why = quality_gate(rep["n_final"], rep["n_raw_items"], len(done), a.min_items_per_100, a.min_kept_ratio)
    if why:
        return _gate_fail(out, why, rep, len(done), "at the end")
    receipt = {"schema": "ens.h60.qa_synth.v1", "recipe": RECIPE, "label": LABEL, **gen_info,
               "corpus_hashes": hashes, "questions_sha12": a.questions_sha12, "args": vars(a) | {"corpus": None,
               "questions": None, "out": Path(a.out).name}, "qa_sha16": sha16(out / "qa.jsonl"), **rep,
               "synth_seconds_this_launch": round(time.time() - t0, 1), **C.source_revision(), **C.host_record()}
    C.write_json(out / "synth_receipt.json", receipt)
    say("synth done:", json.dumps({k: receipt[k] for k in ("n_final", "by_type", "leakage")}))
    return 0


def filter_all(a, tok, docs, train_ids, held_ids, done, xdone, pairs, eval_texts):
    """Filter, dedupe, leakage check -> (kept items, counts without text)."""
    reasons = collections.Counter()
    items, n_raw = [], 0
    for d, row in sorted(done.items()):
        assert_trained_only([d], train_ids, held_ids)
        wins, _ = doc_windows(tok, docs[d], a.window_tokens)
        for it in row["items"]:
            n_raw += 1
            src = tok.t.decode(wins[it["window"]]) if it.get("window") is not None and it["window"] < len(wins) else ""
            why = check_item(it, docs[d].get("title") or "", src + "\n" + (docs[d].get("title") or ""),
                             a.max_answer_words)
            if why:
                reasons[why] += 1
                continue
            items.append({"doc_id": d, "type": it["type"], "q": it["q"], "a": it["a"]})
    ents = {f"{d1}|{d2}": e for d1, d2, e in pairs}
    for key, row in sorted(xdone.items()):
        d1, d2 = row["doc_ids"]
        assert_trained_only([d1, d2], train_ids, held_ids)
        e = ents.get(key, "")
        src = (excerpt_around(tok, docs[d1]["text"], e, a.cross_tokens) + "\n" +
               excerpt_around(tok, docs[d2]["text"], e, a.cross_tokens)) if e else docs[d1]["text"] + docs[d2]["text"]
        for it in row["items"]:
            n_raw += 1
            why = check_item(it, (docs[d1].get("title") or "") + " " + (docs[d2].get("title") or ""), src,
                             a.max_answer_words)
            if why:
                reasons[why] += 1
                continue
            items.append({"doc_id": d1, "doc_ids": [d1, d2], "type": "crossdoc", "q": it["q"], "a": it["a"]})
    items, n_dup = dedupe(items)
    items, leak = leakage_filter(items, eval_texts, a.leak_threshold, a.leak_fact_threshold)
    per_doc = collections.Counter(r["doc_id"] for r in items)
    return items, {"n_documents": len(done), "n_pairs": len(xdone), "n_raw_items": n_raw,
                   "dropped_by_filter": dict(reasons), "dropped_duplicates": n_dup, "leakage": leak,
                   "n_final": len(items), "by_type": dict(collections.Counter(r["type"] for r in items)),
                   "kept_ratio": round(len(items) / n_raw, 4) if n_raw else None,
                   "items_per_document": {"mean": round(len(items) / max(1, len(per_doc)), 2),
                                          "docs_with_items": len(per_doc),
                                          "docs_without_items": len(done) - len(per_doc)}}


def coverage(a, tok, docs, train_ids, done, items, rec):
    """Counts only (qids as sha256[:12]): trained-document eval answers that appear, normalised, in the synthesized
    windows of their document, and in the kept QA answers of their document."""
    by_doc = collections.defaultdict(list)
    for it in items:
        for d in it.get("doc_ids") or [it["doc_id"]]:
            by_doc[d].append(it["a"])
    rows = [r for r in rec if r.get("doc_id") in train_ids and r.get("answer")]
    in_win, in_qa, n_synth = [], [], 0
    for r in rows:
        d = r["doc_id"]
        if d not in done:
            continue
        n_synth += 1
        wins, _ = doc_windows(tok, docs[d], a.window_tokens)
        text = "\n".join(tok.t.decode(wins[w]) for w in done[d].get("windows", []) if w < len(wins))
        if E.correct(text, r["answer"], r.get("aliases")):
            in_win.append(hid(r["qid"]))
        if any(E.correct(x, r["answer"], r.get("aliases")) for x in by_doc.get(d, ())):
            in_qa.append(hid(r["qid"]))
    return {"n_trained_eval_questions": len(rows), "n_with_synthesized_document": n_synth,
            "answer_in_windows": len(in_win), "answer_in_qa_answers": len(in_qa),
            "covered_qid_sha12": sorted(in_qa), "covered_definition": "answer (normalised, corpus_tile_eval.correct) "
                                                                     "in a kept QA answer of the same document"}


def finalize(a, tok, docs, train_ids, held_ids, done, xdone, pairs, eval_texts, out: Path, rec=()) -> dict:
    """filter_all -> qa.jsonl; returns the counts and coverage (no text)."""
    items, counts = filter_all(a, tok, docs, train_ids, held_ids, done, xdone, pairs, eval_texts)
    for i, it in enumerate(items):
        it["item_id"] = f"{hid(it['doc_id'])}-{i:06d}"
    tmp = out / "qa.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in items))
    tmp.replace(out / "qa.jsonl")
    return counts | {"coverage": coverage(a, tok, docs, train_ids, done, items, rec)}


def synth(a) -> int:
    out = Path(a.out)
    if (out / "synth_receipt.json").exists():
        say("already synthesized")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    args_p = out / "args.json"
    keep = {k: v for k, v in vars(a).items() if k not in ("max_seconds", "no_memory_gate", "gate_gib")}
    if args_p.exists() and json.loads(args_p.read_text()) != keep:
        raise SystemExit("resume: synth arguments differ from the first launch")
    C.write_json(args_p, keep)
    docs, train_ids, held_ids, hashes = load_inputs(a.corpus)
    rec, xd = load_questions(a.questions)
    a.questions_sha12 = questions_sha12(Path(a.questions).expanduser())
    if not a.no_memory_gate:
        PB.memory_gate(gib=a.gate_gib)
    mx.set_cache_limit(int(PB.CACHE_LIMIT_GIB * 2 ** 30))
    gen = Generator(a.generator, a.backend)
    info = {"generator_checkpoint": gen.checkpoint, "generator_backend": gen.backend,
            "mlx_lm_version": gen.mlx_lm_version, "decoding": "greedy, chat format, thinking off"}
    say(f"generator {gen.checkpoint} via {gen.backend}")
    check_doc_split(rec, train_ids, held_ids)
    return run_synth(a, gen, gen.tok, docs, train_ids, held_ids, eval_question_texts(rec, xd), hashes, out, info,
                     rec=rec)


# ================================================================== the memory model
class MemoryAdapter(nn.Module):
    """LoRA on all seven projections of every block (lm_bus.LoRA: B zero-initialised, so the untrained memory is the
    base model exactly)."""

    def __init__(self, cfg: Q.Qwen3Config, rank=64, alpha=None):
        super().__init__()
        d, H, K, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        dims = {"q_proj": (d, H * hd), "k_proj": (d, K * hd), "v_proj": (d, K * hd), "o_proj": (H * hd, d),
                "gate_proj": (d, cfg.intermediate_size), "up_proj": (d, cfg.intermediate_size),
                "down_proj": (cfg.intermediate_size, d)}
        self.rank, self.alpha = rank, float(alpha or rank)
        self.lora = {f"b{i}": {n: LB.LoRA(a, b, rank, self.alpha) for n, (a, b) in dims.items()}
                     for i in range(cfg.num_hidden_layers)}


def hidden(model, tokens, adapter=None):
    h = model.model.embed_tokens(tokens)
    for i, blk in enumerate(model.model.layers):
        h = blk(h, "causal", adapter.lora[f"b{i}"] if adapter is not None else None)
    return model.model.norm(h)


def head(model, hn):
    return model.lm_head(hn) if "lm_head" in model else model.model.embed_tokens.as_linear(hn)


def mem_forward(model, tokens, tile=None, zero_values=False):
    """corpus_tile.forward's contract: (logits, None); tile = a MemoryAdapter or None (base); zero_values = base."""
    return head(model, hidden(model, tokens, None if zero_values else tile)), None


def encode_example(tok, q, a):
    """(prompt ids, target ids): the eval prompt (chat, empty think block, " Answer briefly.") and the answer +
    <|im_end|>; only the targets carry loss."""
    end = tok.t.token_to_id("<|im_end|>")
    return E.prompt_ids(tok, q), text_ids(tok, a) + [end]


def stack(examples, pad, bucket=16):
    """[(prompt, target)] -> tokens (B, L), targets (B, L), loss mask (B, L) python lists; position i predicts i+1."""
    L = max(len(p) + len(t) for p, t in examples)
    L = -(-L // bucket) * bucket
    T, Y, W = [], [], []
    for p, t in examples:
        seq = p + t
        T.append(seq + [pad] * (L - len(seq)))
        Y.append((seq[1:] + [pad] * (L - len(seq) + 1))[:L])
        W.append(([0] * (len(p) - 1) + [1] * len(t) + [0] * (L - len(seq) + 1))[:L])
    return T, Y, W


def answer_loss(model, adapter, T, Y, W):
    """Mean CE over the masked (answer) positions only; the head runs on those positions alone."""
    idx = [i for i, w in enumerate(x for row in W for x in row) if w]
    hn = hidden(model, mx.array(T), adapter)
    hs = hn.reshape(-1, hn.shape[-1])[mx.array(idx)]
    y = mx.array([x for row in Y for x in row])[mx.array(idx)]
    ce = nn.losses.cross_entropy(head(model, hs).astype(mx.float32), y, reduction="sum")
    return ce / len(idx), len(idx)


def mean_nll(model, adapter, examples, pad, batch=16):
    tot, n = 0.0, 0
    for i in range(0, len(examples), batch):
        T, Y, W = stack(examples[i:i + batch], pad)
        loss, k = answer_loss(model, adapter, T, Y, W)
        tot += float(loss.item()) * k
        n += k
    return tot / max(1, n)


def split_rows(rows, val_frac, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    k = max(1, int(round(len(rows) * val_frac))) if rows else 0
    val = set(order[:k])
    return [r for i, r in enumerate(rows) if i not in val], [r for i, r in enumerate(rows) if i in val]


def step_order(n, batch, epochs, seed):
    """Row indices for every step: a fresh seeded permutation per epoch (so the order is a function of the step)."""
    steps = []
    for ep in range(epochs):
        p = list(range(n))
        random.Random(seed * 1000 + ep).shuffle(p)
        steps += [p[i:i + batch] for i in range(0, n, batch)]
    return steps


def track_sets(tok, rec, train_ids, held_ids):
    ex = {"train": [], "held": []}
    for r in rec:
        if not r.get("question") or not r.get("answer"):
            continue
        s = "train" if r["doc_id"] in train_ids else "held" if r["doc_id"] in held_ids else None
        if s:
            ex[s].append(encode_example(tok, r["question"], r["answer"]))
    return ex


def tracked(model, adapter, val_ex, track, pad, base=None):
    r = {"val_qa_nll": round(mean_nll(model, adapter, val_ex, pad), 4),
         "evalq_nll_train": round(mean_nll(model, adapter, track["train"], pad), 4),
         "evalq_nll_held": round(mean_nll(model, adapter, track["held"], pad), 4)}
    if base:
        gt = base["evalq_nll_train"] - r["evalq_nll_train"]
        gh = base["evalq_nll_held"] - r["evalq_nll_held"]
        r |= {"evalq_gain_train": round(gt, 4), "evalq_gain_held": round(gh, 4),
              "evalq_gain_trained_minus_held": round(gt - gh, 4)}
    return r


_TRAIN_FREE = {"max_seconds": None, "no_memory_gate": None, "gate_gib": None}


def run_train(a, model, base_ckpt, tok, rows, rec, train_ids, held_ids, out: Path, synth_info: dict) -> int:
    """The training loop on an injected model (tests pass a tiny one). Exit 3 = clean, resumable stop; exit 4 =
    refused (no QA rows or no steps) or non-finite loss."""
    if (out / "train_receipt.json").exists():
        say("already trained")
        return 0
    if not rows:
        say("train: no QA rows; refusing")
        return 4
    resuming = PB.resume_dir(out) is not None
    if not resuming:
        startup = {"args.json", "best.safetensors", "best.tmp.safetensors"}   # a launch stopped before its first save
        if out.exists() and any(p.name not in startup for p in out.iterdir()):
            raise SystemExit(f"refusing nonempty evidence directory without resume state: {out}")
        out.mkdir(parents=True, exist_ok=True)
        for n in startup - {"args.json"}:
            (out / n).unlink(missing_ok=True)
        C.write_json(out / "args.json", vars(a))
    elif json.loads((out / "args.json").read_text()) | _TRAIN_FREE != vars(a) | _TRAIN_FREE:
        raise SystemExit("resume: arguments differ")
    for r in rows:
        assert_trained_only(r.get("doc_ids") or [r["doc_id"]], train_ids, held_ids)
    tr_rows, val_rows = split_rows(rows, a.val_frac, a.seed)
    tr_ex = [encode_example(tok, r["q"], r["a"]) for r in tr_rows]
    val_ex = [encode_example(tok, r["q"], r["a"]) for r in val_rows][:a.val_cap]
    track = track_sets(tok, rec, train_ids, held_ids)
    order = step_order(len(tr_ex), a.batch, a.epochs, a.seed)
    steps = len(order) if not a.steps else min(a.steps, len(order))
    if not tr_ex or steps <= 0:
        say(f"train: nothing to train ({len(tr_ex)} training rows, {steps} steps); refusing")
        return 4
    mx.random.seed(a.seed)
    adapter = MemoryAdapter(model.config, a.rank, a.alpha)
    wu = max(1, int(round(steps * a.warmup)))
    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(a.lr * 0.01, a.lr, wu), optim.cosine_decay(a.lr, max(1, steps - wu), a.lr * 0.1)], [wu]),
        weight_decay=0.0)
    opt.init(adapter.trainable_parameters())
    it, log, t_used = 0, [], 0.0
    if resuming:
        it, log, t_used = PB.load_state(PB.resume_dir(out), adapter, opt, [])
        say(f"resumed at {it}")
    base = next((e for e in log if e.get("step") == 0), None)
    if base is None:
        base = {"step": 0, **tracked(model, adapter, val_ex, track, tok.pad)}
        log.append(base)
        say(json.dumps(base))
    best = min((e for e in log if "val_qa_nll" in e and e["step"] > 0), key=lambda e: e["val_qa_nll"], default=None)
    vg = nn.value_and_grad(adapter, lambda ad, T, Y, W: answer_loss(model, ad, T, Y, W)[0])
    guard = None if a.no_memory_gate else PB.SwapGuard()
    t0, it0, run_loss = time.time(), it, []
    _STOP["requested"] = False
    old = signal.signal(signal.SIGTERM, _on_term)
    try:
        while it < steps:
            swap = guard.check(it - it0) if guard and ((it - it0) <= PB.SWAP_WARMUP_STEPS or it % 10 == 0) else None
            if _STOP["requested"] or swap or (a.max_seconds is not None and time.time() - t0 > a.max_seconds):
                if swap:
                    log.append({"step": it, "swap_stop": swap})
                PB.save_state(out, adapter.parameters(), opt, it, [], log, t_used + time.time() - t0)
                say(f"stopped at {it} (resume state saved)")
                return 3
            T, Y, W = stack([tr_ex[i] for i in order[it]], tok.pad)
            loss, g = vg(adapter, T, Y, W)
            g, gnorm = optim.clip_grad_norm(g, a.clip)
            opt.update(adapter, g)
            mx.eval(adapter.parameters(), opt.state)
            it += 1
            lv = float(loss.item())
            if not math.isfinite(lv):
                say(f"non-finite loss at step {it}")
                return 4
            run_loss.append(lv)
            if it % a.track_every == 0 or it == steps:
                e = {"step": it, "epoch": round(it * a.batch / max(1, len(tr_ex)), 3),
                     "train_loss": round(sum(run_loss) / len(run_loss), 4), "grad_norm": round(float(gnorm.item()), 3),
                     "lr": round(float(opt.learning_rate.item()), 8),
                     **tracked(model, adapter, val_ex, track, tok.pad, base),
                     "elapsed_s": round(t_used + time.time() - t0, 1), **PB.mem_stats()}
                run_loss = []
                log.append(e)
                say(json.dumps(e))
                if best is None or e["val_qa_nll"] < best["val_qa_nll"]:
                    best = e
                    tmp = out / "best.tmp.safetensors"
                    mx.save_safetensors(str(tmp), dict(tree_flatten(adapter.parameters())))
                    tmp.replace(out / "best.safetensors")
            if a.save_every and it % a.save_every == 0 and it < steps:
                PB.save_state(out, adapter.parameters(), opt, it, [], log, t_used + time.time() - t0)
        PB.save_state(out, adapter.parameters(), opt, it, [], log, t_used + time.time() - t0)
    finally:
        signal.signal(signal.SIGTERM, old)
    wpath = out / "adapter.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(adapter.parameters())))
    tag = f"ens-h60-qamemory-{a.model.split('/')[-1].lower()}-r{a.rank}-s{a.seed}"
    receipt = {"schema": "ens.h60.qa_memory_train.v1", "recipe": RECIPE, "label": LABEL, "base_checkpoint": base_ckpt,
               "memory_checkpoint": f"{tag}@{sha16(wpath)[:12]}",
               "memory_checkpoint_best": f"{tag}-best@{sha16(out / 'best.safetensors')[:12]}"
                                         if (out / "best.safetensors").exists() else None,
               "best_step": best["step"] if best else None, **synth_info,
               "adapter": {"kind": "LoRA", "rank": a.rank, "alpha": adapter.alpha, "targets": "q k v o gate up down",
                           "layers": "all"},
               "trainable_params": C.n_params(adapter, trainable=True), "n_train_rows": len(tr_ex),
               "n_val_rows": len(val_ex), "n_track": {k: len(v) for k, v in track.items()}, "steps": steps,
               "schedule": f"AdamW lr {a.lr}, warmup {a.warmup:.0%} linear, cosine to 0.1x, clip {a.clip}, wd 0",
               "loss": "answer tokens + <|im_end|>, chat format with empty think block, prompt as corpus_tile_eval",
               "args": vars(a) | {"corpus": None, "questions": None, "synth": Path(a.synth).name, "out": out.name},
               "log": log, "train_seconds": round(t_used + time.time() - t0, 1), **C.source_revision(),
               **C.host_record()}
    C.write_json(out / "train_receipt.json", receipt)
    say("trained", receipt["memory_checkpoint"])
    return 0


def train(a) -> int:
    out = Path(a.out)
    if (out / "train_receipt.json").exists():
        say("already trained")
        return 0
    sdir = Path(a.synth)
    srec = json.loads((sdir / "synth_receipt.json").read_text())
    if sha16(sdir / "qa.jsonl") != srec["qa_sha16"]:
        raise SystemExit("qa.jsonl differs from its synth receipt")
    docs, train_ids, held_ids, hashes = load_inputs(a.corpus)
    if hashes != srec["corpus_hashes"]:
        raise SystemExit("corpus files differ from the synth receipt")
    rec, _ = load_questions(a.questions)
    rows = [json.loads(l) for l in open(sdir / "qa.jsonl")]
    if not rows:
        say("train: qa.jsonl is empty; refusing")
        return 4
    check_doc_split(rec, train_ids, held_ids)
    if not a.no_memory_gate:
        PB.memory_gate(gib=a.gate_gib)
    mx.set_cache_limit(int(PB.CACHE_LIMIT_GIB * 2 ** 30))
    path = Q.snapshot_dir(a.model)
    model, _cfg, ckpt = Q.load(path, dtype=mx.bfloat16, repo=a.model)
    model.freeze()
    info = {"synth_qa_sha16": srec["qa_sha16"], "generator_checkpoint": srec["generator_checkpoint"],
            "synth_coverage": srec.get("coverage"),
            "corpus_hashes": hashes, "questions_sha12": questions_sha12(Path(a.questions).expanduser())}
    return run_train(a, model, ckpt, LB.Tok(path), rows, rec, train_ids, held_ids, out, info)


# ================================================================== BM25 (pure Python)
def bm25_terms(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOP]


class BM25:
    def __init__(self, passages_terms, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(passages_terms)
        self.len = [len(p) for p in passages_terms]
        self.avg = sum(self.len) / max(1, self.N)
        self.post = collections.defaultdict(list)
        for i, p in enumerate(passages_terms):
            for t, c in collections.Counter(p).items():
                self.post[t].append((i, c))
        self.idf = {t: math.log(1 + (self.N - len(ps) + 0.5) / (len(ps) + 0.5)) for t, ps in self.post.items()}

    def search(self, query_terms, k):
        sc = collections.defaultdict(float)
        for t in set(query_terms):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in self.post[t]:
                sc[i] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [i for i, _ in heapq.nlargest(k, sc.items(), key=lambda x: (x[1], -x[0]))]


def build_passages(tok, docs, ids, n_tok):
    """Non-overlapping n_tok windows of the documents in ids (titles indexed, not pasted): [(doc_id, text, terms)]."""
    out = []
    for d in sorted(ids):
        t = text_ids(tok, docs[d].get("text") or "")
        for j in range(0, len(t), n_tok):
            txt = tok.t.decode(t[j:j + n_tok])
            out.append((d, txt, bm25_terms((docs[d].get("title") or "") + " " + txt)))
    return out


def parse_bm25(spec):
    out = []
    for s in spec.split(","):
        k, p = s.lower().split("x")
        out.append((f"bm25_{int(k)}x{int(p)}", int(k), int(p)))
    return out


def context_prompt(tok, passages, q):
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return chat_ids(tok, f"Passages:\n{ctx}\n\n{q} Answer briefly.")


# ================================================================== bus hook (H60 core-queries-memory arm: later)
SPAN_OPEN, SPAN_CLOSE = "÷÷÷", "÷÷÷"


class MemoryEndpoint:
    """What the core will call over the bus: ask(questions) -> answers in text. The core-queries-memory arm (frozen
    Qwen3-4B asking in text, reading the answer as an H57 delimited span) plugs in here; this eval only uses ask()."""

    def __init__(self, model, tok, adapter, max_new=40, batch=8, gen=None):
        self.model, self.tok, self.adapter, self.max_new, self.batch = model, tok, adapter, max_new, batch
        self.gen = gen or (lambda prompts: cached_generate(model, tok, prompts, max_new, lora=adapter.lora))

    def ask(self, questions):
        out = []
        for i in range(0, len(questions), self.batch):
            out += self.gen([E.prompt_ids(self.tok, q) for q in questions[i:i + self.batch]])
        return out

    @staticmethod
    def span(answer):
        """The H57 delimited message the core would read (delimiters escaped out of the answer)."""
        return f"{SPAN_OPEN} {answer.replace(SPAN_OPEN, '').strip()} {SPAN_CLOSE}"


# ================================================================== eval
def recall_items(outputs_path: Path, prefix: str, rec):
    """[{qid, field, doc, split, arms: {arm: {lenient, strict}}}] from the committed recall outputs."""
    by_q = {r["qid"]: r for r in rec}
    items = {}
    if outputs_path.exists():
        for line in outputs_path.read_text().splitlines():
            o = json.loads(line)
            if not o["cell"].startswith(prefix + "recall.") or o["qid"] not in by_q:
                continue
            r = by_q[o["qid"]]
            it = items.setdefault((o["qid"], o["field"]), {"qid": o["qid"], "field": o["field"], "doc": r["doc_id"],
                                                           "split": r["doc_split"], "arms": {}})
            it["arms"][o["arm"]] = {"lenient": bool(o["lenient"]), "strict": bool(o["strict"])}
    return list(items.values())


def _doc_boot(xs, value, n_boot, rng):
    """Document-level bootstrap of the item mean of value(x): (point, draws (n_boot,))."""
    docs = sorted({x["doc"] for x in xs})
    di = {d: i for i, d in enumerate(docs)}
    sums, cnt = np.zeros(len(docs)), np.zeros(len(docs))
    for x in xs:
        sums[di[x["doc"]]] += value(x)
        cnt[di[x["doc"]]] += 1
    W = np.stack([np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs)) for _ in range(n_boot)])
    return sums.sum() / cnt.sum(), (W @ sums) / np.maximum(W @ cnt, 1)


def _verdict(est, draws, margin):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    se = float(np.std(draws, ddof=1))
    v = "kill" if hi <= margin else "pass" if lo > margin else "undecided"
    return {"estimate": round(float(est), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "se": round(se, 4), "mde80": round(2.8 * se, 4), "margin": margin, "verdict": v}


def kill_measure(items, bm25_arm, bm25_margin=0.0, gap_margin=0.0, n_boot=2000, seed=0):
    """Memory vs BM25 + base on trained documents (paired per item) and the memory's trained-minus-held gap over base
    (difference in differences), each with a document-level paired bootstrap 95% CI (documents resampled within each
    split, a document's questions and fields kept together; all arms on the same draw) and mde80 = 2.8 x se.
    A test kills only when its CI excludes a positive effect (upper bound <= margin), passes when the lower bound is
    above the margin, and is otherwise undecided. A missing arm or split is undecided, never 'not killed'."""
    tr = [x for x in items if x["split"] == "train"]
    he = [x for x in items if x["split"] == "held_out"]
    why = []
    if not tr:
        why.append("no trained-document recall items")
    if not he:
        why.append("no held-document recall items")
    for arm, xs in (("base", tr + he), ("memory", tr + he), (bm25_arm, tr)):
        if xs and not all(arm in x["arms"] for x in xs):
            why.append(f"arm {arm} missing on some items")
    meta = {"bm25_arm": bm25_arm, "bm25_margin": bm25_margin, "gap_margin": gap_margin, "decided_on": "lenient",
            "n_boot": n_boot, "n_items": {"train": len(tr), "held_out": len(he)},
            "n_docs": {"train": len({x["doc"] for x in tr}), "held_out": len({x["doc"] for x in he})}}
    if why:
        return meta | {"decision": "undecided", "killed": None, "reasons": why}
    out = {}
    for score in ("lenient", "strict"):
        def v(arm, x, score=score):
            return float(x["arms"][arm][score])
        rng = np.random.default_rng(seed)
        e_bm, d_bm = _doc_boot(tr, lambda x: v("memory", x) - v(bm25_arm, x), n_boot, rng)
        rng = np.random.default_rng(seed + 1)
        e_mt, d_mt = _doc_boot(tr, lambda x: v("memory", x) - v("base", x), n_boot, rng)
        e_mh, d_mh = _doc_boot(he, lambda x: v("memory", x) - v("base", x), n_boot, rng)
        mean = lambda xs, arm: round(sum(v(arm, x) for x in xs) / len(xs), 4)  # noqa: E731
        t_bm, t_gap = _verdict(e_bm, d_bm, bm25_margin), _verdict(e_mt - e_mh, d_mt - d_mh, gap_margin)
        verdicts = (t_bm["verdict"], t_gap["verdict"])
        out[score] = {"memory_trained": mean(tr, "memory"), "memory_held": mean(he, "memory"),
                      "base_trained": mean(tr, "base"), "base_held": mean(he, "base"),
                      f"{bm25_arm}_trained": mean(tr, bm25_arm),
                      "memory_minus_bm25_trained": t_bm, "did_vs_base": t_gap,
                      "decision": "killed" if "kill" in verdicts else "not_killed" if verdicts == ("pass", "pass")
                      else "undecided"}
    d = out["lenient"]["decision"]
    return meta | out | {"decision": d, "killed": True if d == "killed" else False if d == "not_killed" else None}


def run_eval(a, model, tok, adapter, docs, train_ids, rec, xd, qhash, run: Path, gen=None, held_ids=None,
             mem_checkpoint="", covered=None) -> dict:
    """All cells, resumable (EvalCells; outputs committed with their cell). gen(model, prompts, lora, impl) -> texts.
    covered: sha256[:12] qids of trained questions covered by the synthesized QA (synth coverage), or None."""
    from research.ens import readout_check as R
    if held_ids is not None:
        check_doc_split(rec, train_ids, held_ids)
    if gen is None:
        def gen(m, P, lo, impl):
            if impl == "reference":
                return E.generate(m, tok, lo, P, max_new=a.max_new, fwd=mem_forward)
            return cached_generate(m, tok, P, a.max_new, lora=lo.lora if lo else None)
    bm25s = parse_bm25(a.bm25) if a.bm25 else []
    index = {}
    for name, k, p in bm25s:
        if p not in index:
            ps = build_passages(tok, docs, train_ids, p)
            index[p] = (ps, BM25([x[2] for x in ps]))
            say(f"bm25: {len(ps)} passages of {p} tokens over the trained documents")
    arms = ["base", "memory"] + [n for n, _, _ in bm25s]
    decoders = {arm: (a.base_gen if arm == "base" else a.gen_impl) for arm in arms}
    key = json.dumps([arms, a.max_new, decoders, mem_checkpoint.rpartition("@")[2], a.which])
    prefix = f"{qhash}.h60.{hashlib.sha256(key.encode()).hexdigest()[:8]}."
    cells = PB.EvalCells(run, time.time() + a.max_seconds if a.max_seconds else None)
    outp = run / f"outputs-{prefix.rstrip('.')}.jsonl"
    if outp.exists():                                          # keep only rows of committed cells
        kept = [l for l in outp.read_bytes().splitlines(keepends=True)
                if l.endswith(b"\n") and cells.done(json.loads(l)["cell"])]
        outp.write_bytes(b"".join(kept))
    outs = outp.open("a")
    syn = [r for r in xd if r.get("type") == "synthesis"]
    strict = R.Strict(tok, rec)            # recall rows only, as readout_check (H56 comparability)

    def prompts_for(arm, rows, field):
        if arm in ("base", "memory"):
            return [E.prompt_ids(tok, r[field]) for r in rows], [0] * len(rows), [None] * len(rows)
        _, k, p = next(x for x in bm25s if x[0] == arm)
        ps, bm = index[p]
        P, cost, hits = [], [], []
        for r in rows:
            top = bm.search(bm25_terms(r[field]), k)
            P.append(context_prompt(tok, [ps[i][1] for i in top], r[field]))
            cost.append(sum(len(text_ids(tok, ps[i][1])) for i in top))
            want = set(r.get("doc_ids") or ([r["doc_id"]] if r.get("doc_id") else []))
            hits.append(bool(want & {ps[i][0] for i in top}) if want else None)
        return P, cost, hits

    def gen_cell(key, rows, field):
        key = prefix + key
        if cells.done(key) or not rows:
            return
        res = {}
        for arm in arms:
            len_tot = cost_tot = 0
            lenient = strict_n = 0
            hit = [0, 0]
            for i in range(0, len(rows), a.batch):
                cells.check()
                sub = rows[i:i + a.batch]
                P, cost, hits = prompts_for(arm, sub, field)
                texts = gen(model, P, adapter if arm == "memory" else None, decoders[arm])
                for r, o, c, h in zip(sub, texts, cost, hits):
                    ok, st = bool(E.correct(o, r["answer"], r.get("aliases"))), bool(strict(o, r))
                    n_out = len(tok.encode(o))
                    lenient += ok
                    strict_n += st
                    len_tot += n_out
                    cost_tot += c + (n_out if arm == "memory" else 0)
                    if h is not None:
                        hit[0] += h
                        hit[1] += 1
                    outs.write(json.dumps({"cell": key, "qid": r["qid"], "arm": arm, "field": field, "output": o,
                                           "lenient": ok, "strict": st}) + "\n")
            n = len(rows)
            res[arm] = {"lenient": round(lenient / n, 4), "strict": round(strict_n / n, 4),
                        "out_tok_mean": round(len_tot / n, 2), "context_tok_mean": round(cost_tot / n, 2)}
            if hit[1]:
                res[arm]["retrieved_source_doc"] = round(hit[0] / hit[1], 4)
        outs.flush()
        cells.put(key, res | {"n": len(rows)})

    for split in ("train", "held_out"):
        rows = [r for r in rec if r.get("doc_split") == split]
        for field in ("question", "rephrase"):
            gen_cell(f"recall.{split}.{field}", rows, field)
    graph = [r for r in xd if r.get("type") == "graph" and r.get("choices")]
    for mix in ("all_train", "has_held_out", "all_held_out"):
        rows = [r for r in syn if r.get("split_mix") == mix]
        for field in ("question", "rephrase"):
            gen_cell(f"synthesis.{mix}.{field}", rows, field)
        grows = [r for r in graph if r.get("split_mix") == mix]
        for field in ("question", "rephrase"):
            key = f"{prefix}graph_mc.{mix}.{field}"
            if cells.done(key) or not grows:
                continue
            tally = collections.Counter()
            for r in grows:
                cells.check()
                for arm in arms:
                    lo = adapter if arm == "memory" else None
                    if arm in ("base", "memory"):
                        pq, pn = E.prompt_ids(tok, r[field]), tok.encode(chat_prompt(E.NEUTRAL))
                    else:      # the neutral prompt carries the same passages, so their strings earn no credit
                        _, k, p = next(x for x in bm25s if x[0] == arm)
                        ps, bm = index[p]
                        pas = [ps[i][1] for i in bm.search(bm25_terms(r[field]), k)]
                        pq, pn = context_prompt(tok, pas, r[field]), context_prompt(tok, pas, E.NEUTRAL)
                    la = E._choice_ll(model, tok, lo, False, pq, r["choices"], mem_forward)
                    lb = E._choice_ll(model, tok, lo, False, pn, r["choices"], mem_forward)
                    sc = [x - y for x, y in zip(la, lb)]
                    pick = r["choices"][max(range(len(sc)), key=sc.__getitem__)]
                    tally[arm] += int(pick == r["choice_answer"])
                    outs.write(json.dumps({"cell": key, "qid": r["qid"], "arm": arm, "pick": pick,
                                           "correct": pick == r["choice_answer"]}) + "\n")
            outs.flush()
            cells.put(key, {arm: round(tally[arm] / len(grows), 4) for arm in arms} |
                      {"n": len(grows), "chance": round(sum(1 / len(r["choices"]) for r in grows) / len(grows), 4)})
    outs.close()
    mine = {k[len(prefix):]: v for k, v in cells.cells.items() if k.startswith(prefix)}
    items = recall_items(outp, prefix, rec)
    karm = a.kill_bm25 if a.kill_bm25 in arms else (bm25s[0][0] if bm25s else "none")
    kill = kill_measure(items, karm, a.bm25_margin, a.gap_margin, a.n_boot)
    if covered is None:
        kcov = {"decision": "undecided", "killed": None, "reasons": ["no synth coverage record"]}
    else:
        cov = set(covered)
        kcov = kill_measure([x for x in items if x["split"] != "train" or hid(x["qid"]) in cov], karm,
                            a.bm25_margin, a.gap_margin, a.n_boot)
        kcov["restriction"] = "trained questions whose answer is in a kept QA answer of their document; held: all"
    return {"cells": mine, "arms": arms, "decoders": decoders, "cell_prefix": prefix, "outputs_file": outp.name,
            "kill": kill, "kill_covered": kcov}


def evaluate(a) -> int:
    run = Path(a.run)
    if (run / "receipt.json").exists():
        say("already evaluated")
        return 0
    rec_t = json.loads((run / "train_receipt.json").read_text())
    targs = rec_t["args"]
    docs, train_ids, held_ids, hashes = load_inputs(a.corpus)
    if hashes != rec_t["corpus_hashes"]:
        raise SystemExit("corpus files differ from the training receipt")
    qdir = Path(a.questions).expanduser()
    rec, xd = load_questions(qdir)
    qhash = questions_sha12(qdir)
    if not a.no_memory_gate:
        PB.memory_gate(gib=a.gate_gib)
    mx.set_cache_limit(int(PB.CACHE_LIMIT_GIB * 2 ** 30))
    path = Q.snapshot_dir(targs["model"])
    model, _cfg, ckpt = Q.load(path, dtype=mx.bfloat16, repo=targs["model"])
    if ckpt != rec_t["base_checkpoint"]:
        raise SystemExit("base checkpoint differs from the training receipt")
    model.freeze()
    tok = LB.Tok(path)
    adapter = MemoryAdapter(model.config, targs["rank"], targs["alpha"])
    wfile, ck = (("best.safetensors", rec_t["memory_checkpoint_best"]) if a.which == "best"
                 and rec_t.get("memory_checkpoint_best") else ("adapter.safetensors", rec_t["memory_checkpoint"]))
    C.load_checkpoint_weights(adapter, run / wfile, ck)
    adapter.freeze()
    cov = (rec_t.get("synth_coverage") or {}).get("covered_qid_sha12")
    res = run_eval(a, model, tok, adapter, docs, train_ids, rec, xd, qhash, run, held_ids=held_ids,
                   mem_checkpoint=ck, covered=cov)
    receipt = {"schema": "ens.h60.qa_memory_eval.v1", "recipe": RECIPE, "label": LABEL,
               "base_checkpoint": ckpt, "memory_checkpoint": ck, "memory_weights": wfile,
               "generator_checkpoint": rec_t.get("generator_checkpoint"), "synth_qa_sha16": rec_t.get("synth_qa_sha16"),
               "corpus_hashes": hashes, "questions_sha12": qhash,
               "scoring": {"lenient": "corpus_tile_eval.correct over the whole output",
                           "strict": "readout_check.Strict over recall rows (first line, 12 tokens, no other "
                                     "recall answer before it)",
                           "graph_mc": "calibrated choice log-likelihood (question minus neutral prompt; BM25 arms: "
                                       "the neutral prompt carries the same passages)",
                           "generation": f"greedy, max_new {a.max_new}, chat format thinking off; decoder per arm "
                                         "in eval.decoders (reference = corpus_tile_eval.generate, as H56)"},
               "context_cost": "context_tok_mean: BM25 = pasted passage tokens; memory = its answer tokens (what the "
                               "core would read as a span); base = 0",
               "eval": res, "args": vars(a) | {"run": run.name, "questions": qdir.name, "corpus": None},
               "train_receipt_source_revision": rec_t.get("source_revision"), **C.source_revision(),
               **C.host_record()}
    C.write_json(run / "receipt.json", receipt)
    say("eval kill:", res["kill"]["decision"], "| covered:", res["kill_covered"]["decision"])
    return 0


def probe_check(a) -> int:
    """Gate after the probe: held-QA validation loss finite and falling (last tracked < step 0 - min_drop)."""
    r = json.loads((Path(a.run) / "train_receipt.json").read_text()) if (Path(a.run) / "train_receipt.json").exists() \
        else None
    if r is None:
        say("probe: no train receipt (the probe failed or stopped)")
        return 4
    vals = [e["val_qa_nll"] for e in r["log"] if "val_qa_nll" in e]
    losses = [e["train_loss"] for e in r["log"] if "train_loss" in e]
    ok = len(vals) >= 2 and all(math.isfinite(x) for x in vals + losses) and vals[-1] < vals[0] - a.min_drop
    C.write_json(Path(a.run) / "probe_check.json", {"val_qa_nll": vals, "min_drop": a.min_drop, "ok": ok})
    say(f"probe: val_qa_nll {vals[0] if vals else None} -> {vals[-1] if vals else None}: {'ok' if ok else 'NOT falling'}")
    return 0 if ok else 4


def smoke_check(a) -> int:
    """After the --limit-docs smoke synth: counts and two example (q, a) pairs written to --out only (private,
    never printed). Exit 4 unless the smoke kept usable items."""
    sd = Path(a.synth)
    rec_p, fail_p = sd / "synth_receipt.json", sd / "synth_gate_failed.json"
    r = json.loads(rec_p.read_text()) if rec_p.exists() else json.loads(fail_p.read_text()) if fail_p.exists() else {}
    rows = [json.loads(l) for l in open(sd / "qa.jsonl")] if (sd / "qa.jsonl").exists() and rec_p.exists() else []
    ok = rec_p.exists() and len(rows) > 0
    C.write_json(Path(a.out), {"label": LABEL, "ok": ok, "n_raw_items": r.get("n_raw_items"),
                               "n_kept": r.get("n_final"), "kept_ratio": r.get("kept_ratio"),
                               "gate_failure": r.get("reason"), "generator_checkpoint": r.get("generator_checkpoint"),
                               "generator_backend": r.get("generator_backend"),
                               "examples": [{"type": x["type"], "q": x["q"], "a": x["a"]} for x in rows[:2]]})
    say(f"smoke: raw {r.get('n_raw_items')}, kept {r.get('n_final')}: {'ok' if ok else 'NOT usable'}")
    return 0 if ok else 4


# ================================================================== CLI
def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--corpus", required=True)
    s.add_argument("--questions", required=True, help="eval dir (leakage check only)")
    s.add_argument("--generator", default=GENERATOR)
    s.add_argument("--backend", choices=("auto", "mlx_lm", "ens"), default="auto")
    s.add_argument("--items-per-doc", type=int, default=48, help="cap on items asked per document")
    s.add_argument("--items-per-1k", type=float, default=4.0, help="items asked per 1,000 document tokens")
    s.add_argument("--min-items", type=int, default=6)
    s.add_argument("--min-per-call", type=int, default=3)
    s.add_argument("--max-per-call", type=int, default=8)
    s.add_argument("--window-tokens", type=int, default=1536)
    s.add_argument("--cross-tokens", type=int, default=600)
    s.add_argument("--crossdoc-pairs", type=int, default=200)
    s.add_argument("--crossdoc-items", type=int, default=2)
    s.add_argument("--max-answer-words", type=int, default=10)
    s.add_argument("--leak-threshold", type=float, default=0.6, help="drop at all-token Jaccard >= this")
    s.add_argument("--leak-fact-threshold", type=float, default=0.5,
                   help="drop at content-token Jaccard >= this when the answer matches that eval question's")
    s.add_argument("--gen-batch", type=int, default=8)
    s.add_argument("--max-new", type=int, default=560)
    s.add_argument("--sanity-batches", type=int, default=3)
    s.add_argument("--gate-docs", type=int, default=25, help="QA quality gate after this many documents (and at the end)")
    s.add_argument("--min-items-per-100", type=float, default=20.0, help="kept items per 100 trained documents")
    s.add_argument("--min-kept-ratio", type=float, default=0.2)
    s.add_argument("--limit-docs", type=int, default=0, help="smoke runs only")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-seconds", type=float, default=None)
    s.add_argument("--gate-gib", type=float, default=8.0)
    s.add_argument("--no-memory-gate", action="store_true")
    s.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--synth", required=True)
    t.add_argument("--corpus", required=True)
    t.add_argument("--questions", required=True, help="eval dir (answer-NLL tracking only, never trained on)")
    t.add_argument("--model", default=Q.QWEN3_06B)
    t.add_argument("--rank", type=int, default=64)
    t.add_argument("--alpha", type=float, default=None, help="LoRA alpha (default = rank: scale 1)")
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--warmup", type=float, default=0.05)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--steps", type=int, default=0, help="cap (probes); 0 = all epochs")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--val-cap", type=int, default=2000)
    t.add_argument("--track-every", type=int, default=250)
    t.add_argument("--save-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-seconds", type=float, default=None)
    t.add_argument("--gate-gib", type=float, default=12.0)
    t.add_argument("--no-memory-gate", action="store_true")
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--questions", required=True)
    e.add_argument("--corpus", required=True, help="BM25 index over its trained documents")
    e.add_argument("--bm25", default="1x48,3x128", help="arms KxP: k passages of P tokens")
    e.add_argument("--kill-bm25", default="bm25_1x48", help="the matched-cost BM25 arm for the kill")
    e.add_argument("--bm25-margin", type=float, default=0.0)
    e.add_argument("--gap-margin", type=float, default=0.0)
    e.add_argument("--which", choices=("best", "final"), default="best")
    e.add_argument("--gen-impl", choices=("cached", "reference"), default="cached", help="memory and BM25 arms")
    e.add_argument("--base-gen", choices=("cached", "reference"), default="reference",
                   help="base arm (reference = corpus_tile_eval.generate, as H56)")
    e.add_argument("--n-boot", type=int, default=2000)
    e.add_argument("--max-new", type=int, default=E.MAX_NEW)
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--max-seconds", type=float, default=None)
    e.add_argument("--gate-gib", type=float, default=12.0)
    e.add_argument("--no-memory-gate", action="store_true")
    pc = sub.add_parser("probe-check")
    pc.add_argument("--run", required=True)
    pc.add_argument("--min-drop", type=float, default=0.0)
    sc = sub.add_parser("smoke-check")
    sc.add_argument("--synth", required=True)
    sc.add_argument("--out", required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    sys.exit({"synth": synth, "train": train, "eval": evaluate, "probe-check": probe_check,
              "smoke-check": smoke_check}[a.cmd](a))


if __name__ == "__main__":
    main()
