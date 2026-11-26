"""Readout checks on a finished H51 / H56 run (training lessons section 7, 2026-11-26): is document content that a
tile or adapter stored (trained-document NLL gains) answerable through another readout, or not stored at all?

H51 and H56 measured question recall only through the model's own generation path (its tied head). Following the
three-interface diagnosis of continual learning (arXiv 2611.30151: most apparent forgetting was readout mismatch), two
cheap checks run on the finished run, per arm (base = no component; the run's component = its tile, or the LoRA of
the H51 lora-only run; optionally the H51 tile with its memory emptied):

1. Realigned readout. Everything frozen (core, tile, LoRA). Only a rank-r readout is trained on top of the final-norm
   hidden state, at the LM head's input: hn -> hn + B(A hn) (A: d -> r, B: r -> d, B zero-initialised, so the
   untrained readout is the native model bit for bit). The delta is computed in float32 and rounded to the state's
   dtype (bf16 on Qwen3) before it is added, in training and in generation alike. The tied head stays frozen.
   Training data: question -> answer pairs from eval/recall.jsonl whose source documents are in D_fit, a seeded
   random half (by document) of ALL trained documents that have questions (77 documents / 231 questions on the
   pilot corpus: D_fit 38 documents, D_test 39, held 30 documents / 90 questions); the `question` field only
   (--train-fields), so the evaluated `rephrase` phrasings are new to the readout; prompt in chat format with
   thinking off (corpus_tile_eval.prompt_ids); loss on the answer tokens plus the turn end <|im_end|> (teacher
   forced), so the readout also learns to stop. Defaults: 300 steps, AdamW lr 1e-3 (constant, no weight decay),
   batch 8 questions, global gradient-norm clip 1.0, rank 16; the same data, batch order, steps, lr and initial
   readout for every arm. Because the backbone is frozen and the readout sits after the last block, the final-norm
   states at the answer positions are computed once per arm and the readout trains on them. This matches the full
   forward to bf16 tolerance (allclose 2e-2 on the logits, tested), not bit for bit: the cached states come from
   differently padded batches.
   Evaluation: greedy generation (corpus_tile_eval.generate, the same max_new), both fields by default
   (--eval-fields question,rephrase: about 2 x 116 D_test, 2 x 90 held items). Native (no readout) runs on every
   trained and held question in corpus_tile_eval.qa_eval's own order and batching, so it can be checked per qid
   against the run's stored qa_eval outputs (outputs-<qhash>.jsonl, qhash over recall + crossdoc as qa_eval does);
   realigned runs on D_test, held and D_fit (did the readout learn its own training questions).
   Two scores: lenient = corpus_tile_eval.correct over the whole output (the H51 scorer); strict = the answer is in
   the window (the first non-empty line, cut to its first 12 tokens) and no other question's answer or alias
   appears in the window before it (hedged lists of candidates do not score). Output lengths (re-encoded tokens)
   are reported per arm, mode, field and split.
   The split is by document, so the readout cannot learn the D_test facts; the share of D_test and held answers that
   also occur among D_fit answers is reported (an answer prior the readout could learn from D_fit).
   The decisive contrast, per component: lift = realigned - native; lift(component) - lift(base) on D_test and on
   held, and the difference in differences (that on D_test minus that on held), with a document-level paired
   bootstrap 95% CI (documents resampled within each split, a document's questions and both fields kept together,
   all four conditions on the same draw; 2,000 resamples, seeded) and the minimum detectable effect at 80% power,
   two-sided 5% (2.8 x the bootstrap standard error).
2. Classifier-free separability. For trained documents with >= 2 chunks (corpus_tile.doc_chunks), the mean-pooled
   final-norm hidden state over real tokens at positions >= 1 of each chunk, under each arm. Vectors are centred on
   the set mean and L2-normalised (cosine geometry; no labels used). Leave-one-chunk-out nearest-class-mean accuracy
   (the left-out chunk is excluded from its document's mean), kNN-1 accuracy and the silhouette (cosine distance) of
   document identity, for trained documents and, separately, held documents (the control: the component never saw
   them), with component - base and (component - base on trained) - (component - base on held) contrasts. Capped by
   --sep-docs and --sep-chunks (seeded selection). WEAK SUPPORTING EVIDENCE ONLY: the trained chunks are the tile's
   own training windows, so any document-keyed delta (even one that stores no answerable fact) raises their
   separability. It can support a recall result, never replace one.

The receipt (readout_check.v1) carries no corpus text: document ids and qids are written as opaque hashes
(sha256[:12]), run_dir relative to the repository; it is labelled `private_corpus: never publish`. Generated outputs go
to a separate private file next to it (<out stem>.outputs.jsonl, same label) for audit.

  python -m research.ens.readout_check --run <run dir> --kind side_tile|corpus_tile --questions <eval dir> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import statistics
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

SCHEMA = "readout_check.v1"
OUTPUTS_SCHEMA = "readout_check.outputs.v1"
LABEL = "private_corpus: never publish"
PLACEMENT = ("final-norm hidden state (LM head input): hn -> hn + B(A hn), B zero-init, delta computed in float32 and "
             "rounded to the state dtype; tied head frozen")
SEP_CAVEAT = ("weak supporting evidence: trained chunks are the tile's own training windows, so a document-keyed delta "
              "raises separability without answerable facts")
STRICT_TOKENS = 12
STORED_ARM = {"lora": "tile"}          # qa_eval names the lora-only run's adapter arm "tile"
SPLITS = ("d_fit", "d_test", "held")


def hid(x) -> str:
    """Opaque id: sha256(x)[:12], unless x already is a 12-hex id."""
    x = str(x)
    return x if re.fullmatch(r"[0-9a-f]{12}", x) else hashlib.sha256(x.encode()).hexdigest()[:12]


def questions_hash(qdir: Path) -> str:
    """The original qa_eval qhash (runs evaluated before main d52f055d9): recall.jsonl + crossdoc.jsonl bytes."""
    files = [qdir / "recall.jsonl", qdir / "crossdoc.jsonl"]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files if f.exists())).hexdigest()[:12]


def questions_hash_v2(qdir: Path) -> str:
    """qa_eval's qhash since main d52f055d9 ("score2" cells): per-file name + sha256 digest."""
    digest = hashlib.sha256()
    for f in (qdir / "recall.jsonl", qdir / "crossdoc.jsonl"):
        if f.is_file():
            digest.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    return digest.hexdigest()[:12]


def stored_outputs_any(run: Path, qdir: Path):
    """Stored qa_eval recall outputs in either format: score2 (current qa_eval) first, then the original. Returns
    (outputs or None, which format)."""
    h2 = questions_hash_v2(qdir)
    out = stored_outputs(run, h2, file=f"outputs-{h2}.score2.jsonl", prefix=f"{h2}.score2.")
    if out is not None:
        return out, "score2"
    out = stored_outputs(run, questions_hash(qdir))
    return out, ("v1" if out is not None else None)


# ------------------------------------------------------------------ readout
class Readout(nn.Module):
    """hn -> hn + B(A hn). A seeded (explicit key, so every arm starts from the same readout), B zero."""

    def __init__(self, d, rank=16, seed=0):
        super().__init__()
        self.a = nn.Linear(d, rank, bias=False)
        self.b = nn.Linear(rank, d, bias=False)
        self.a.weight = mx.random.normal((rank, d), key=mx.random.key(seed)) / (d ** 0.5)
        self.b.weight = mx.zeros((d, rank))

    def __call__(self, h):
        return h + self.b(self.a(h.astype(mx.float32))).astype(h.dtype)


def lm(base, hn):
    """The LM head without its norm (qwen3.Qwen3.head after model.norm)."""
    return base.lm_head(hn) if "lm_head" in base else base.model.embed_tokens.as_linear(hn)


class Tap:
    """A stand-in for the model that the existing forwards (corpus_tile.forward, side_tile.forward) and
    corpus_tile_eval.generate take unchanged: same blocks, a different head. mode 'norm' returns the final-norm hidden
    state instead of logits; mode 'readout' returns logits through the readout."""

    def __init__(self, base, mode, readout=None):
        self._base, self._mode, self._readout = base, mode, readout
        self.model = base.model
        self.config = base.config

    def head(self, h):
        hn = self.model.norm(h)
        if self._mode == "norm":
            return hn
        return lm(self._base, self._readout(hn))

    def __getattr__(self, k):
        return getattr(self._base, k)


# ------------------------------------------------------------------ splits
def doc_split(rows, train_ids, held_ids, seed):
    """{d_fit, d_test, held}: sorted document ids. D_fit / D_test: a seeded random half of the trained documents that
    have questions; held: the held-out documents that have questions. Refuses rows whose doc_split disagrees with the
    corpus split."""
    for r in rows:
        want = "train" if r["doc_id"] in train_ids else "held_out" if r["doc_id"] in held_ids else None
        if want is None or r.get("doc_split", want) != want:
            raise SystemExit(f"question {hid(r.get('qid'))}: doc_split disagrees with the corpus split")
    tdocs = sorted({r["doc_id"] for r in rows if r["doc_id"] in train_ids})
    order = list(tdocs)
    random.Random(seed).shuffle(order)
    k = len(order) // 2
    return {"d_fit": sorted(order[:k]), "d_test": sorted(order[k:]),
            "held": sorted({r["doc_id"] for r in rows if r["doc_id"] in held_ids})}


def rows_for(rows, ids, cap=None, seed=0):
    """Rows whose document is in ids, in file order (qa_eval's order); cap: a seeded subset, file order kept."""
    ids = set(ids)
    sel = [i for i, r in enumerate(rows) if r["doc_id"] in ids]
    if cap is not None and len(sel) > cap:
        sel = sorted(random.Random(seed).sample(sel, cap))
    return [rows[i] for i in sel]


def answer_overlap(q):
    """Share of D_test / held questions whose answer (or alias), normalised, is also a D_fit answer or alias
    (exact_norm), and whose scorer would accept some D_fit answer string emitted verbatim (fit_answer_scores)."""
    fit = q["d_fit"]
    fit_norm = {E._norm(x) for r in fit for x in [r["answer"]] + list(r.get("aliases") or [])} - {""}
    fit_ans = sorted({r["answer"] for r in fit})
    out = {}
    for name in ("d_test", "held"):
        rs = q[name]
        ex = [any(E._norm(x) in fit_norm for x in [r["answer"]] + list(r.get("aliases") or [])) for r in rs]
        sc = [any(E.correct(fa, r["answer"], r.get("aliases")) for fa in fit_ans) for r in rs]
        out[name] = {"n": len(rs), "exact_norm": round(sum(ex) / len(rs), 4) if rs else None,
                     "fit_answer_scores": round(sum(sc) / len(rs), 4) if rs else None}
    return out


# ------------------------------------------------------------------ states
def end_id(tok):
    i = tok.t.token_to_id("<|im_end|>")
    if i is None:
        raise SystemExit("tokenizer has no <|im_end|>")
    return i


def answer_examples(tok, rows, fields):
    """[(prompt ids, answer ids + <|im_end|>)]."""
    end = end_id(tok)
    out = []
    for r in rows:
        ans = tok.encode(r["answer"])
        for f in fields:
            if r.get(f) and ans:
                out.append((E.prompt_ids(tok, r[f]), ans + [end]))
    return out


def answer_states(model, fwd, tile, zv, examples, pad, batch=8):
    """[(H (n_target, d) final-norm states at the positions that predict the target tokens, Y (n_target,))]."""
    tap = Tap(model, "norm")
    out = []
    for i in range(0, len(examples), batch):
        sub = examples[i:i + batch]
        seqs = [p + a for p, a in sub]
        L = max(len(s) for s in seqs)
        T = mx.array([s + [pad] * (L - len(s)) for s in seqs])
        hn, _ = fwd(tap, T, tile, zero_values=zv)
        for b, (p, a) in enumerate(sub):
            out.append((hn[b, len(p) - 1:len(p) + len(a) - 1], mx.array(a)))
        mx.eval([x for x, _ in out[-len(sub):]])
    return out


def readout_loss(base, ro, H, Y):
    logits = lm(base, ro(H))
    return mx.mean(nn.losses.cross_entropy(logits.astype(mx.float32), Y, reduction="none"))


def train_readout(base, states, a, log_every=50):
    """Train a fresh Readout on cached answer states. Returns (readout, log)."""
    d = base.config.hidden_size
    ro = Readout(d, a.rank, seed=a.seed)
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.0)
    opt.init(ro.trainable_parameters())
    rng = random.Random(20_000 + a.seed)
    vg = nn.value_and_grad(ro, lambda r, H, Y: readout_loss(base, r, H, Y))
    log = []
    for step in range(1, a.steps + 1):
        idx = rng.sample(range(len(states)), a.batch) if len(states) >= a.batch else \
            [rng.randrange(len(states)) for _ in range(a.batch)]
        H = mx.concatenate([states[i][0] for i in idx])
        Y = mx.concatenate([states[i][1] for i in idx])
        loss, g = vg(ro, H, Y)
        if a.clip:
            g, _ = optim.clip_grad_norm(g, a.clip)
        opt.update(ro, g)
        mx.eval(ro.parameters(), opt.state)
        if step == 1 or step % log_every == 0 or step == a.steps:
            log.append({"step": step, "loss": round(float(loss.item()), 4)})
    ro.freeze()
    return ro, log


# ------------------------------------------------------------------ scoring
class Strict:
    """Strict score: the answer is in the window (first non-empty line, first STRICT_TOKENS tokens of it) and no other
    question's answer or alias appears there before it."""

    def __init__(self, tok, rows, n_tok=STRICT_TOKENS):
        self.tok, self.n_tok = tok, n_tok
        self.cands = sorted({n for r in rows for x in [r["answer"]] + list(r.get("aliases") or [])
                             if (n := E._norm(x))})

    def window(self, text):
        lines = [l for l in (text or "").splitlines() if l.strip()]
        return self.tok.t.decode(self.tok.encode(lines[0].strip())[:self.n_tok]) if lines else ""

    @staticmethod
    def _first(text_n, phrases_n):
        best = None
        for p in phrases_n:
            if p and p in text_n:
                m = re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", text_n)
                if m and (best is None or m.start() < best):
                    best = m.start()
        return best

    def __call__(self, text, row):
        w = self.window(text)
        if not E.correct(w, row["answer"], row.get("aliases")):
            return False
        own = {E._norm(x) for x in [row["answer"]] + list(row.get("aliases") or []) + E._parts(row["answer"])} - {""}
        wn = E._norm(w)
        p_true = self._first(wn, own)
        p_other = self._first(wn, [c for c in self.cands if c not in own])
        return p_true is not None and (p_other is None or p_other >= p_true)


def generate_rows(model, tok, fwd, tile, zv, rows, field, max_new, batch, strict):
    """{qid: {output, n_tok, lenient, strict}}, batches of `batch` rows in the given order (qa_eval's batching)."""
    out = {}
    for i in range(0, len(rows), batch):
        sub = rows[i:i + batch]
        prompts = [E.prompt_ids(tok, r[field]) for r in sub]
        gens = E.generate(model, tok, tile, prompts, zero_values=zv, max_new=max_new, fwd=fwd)
        for r, o in zip(sub, gens):
            out[r["qid"]] = {"output": o, "n_tok": len(tok.encode(o)),
                             "lenient": bool(E.correct(o, r["answer"], r.get("aliases"))), "strict": bool(strict(o, r))}
    return out


def recall(model, tok, fwd, tile, zv, rows, field, max_new, batch):
    """{qid: lenient correct} (corpus_tile_eval's scorer)."""
    g = generate_rows(model, tok, fwd, tile, zv, rows, field, max_new, batch, lambda o, r: False)
    return {k: v["lenient"] for k, v in g.items()}


def _acc(vals):
    vals = list(vals)
    return round(sum(vals) / len(vals), 4) if vals else None


def summary(g, max_new):
    n = [v["n_tok"] for v in g.values()]
    return {"n": len(g), "lenient": _acc(v["lenient"] for v in g.values()),
            "strict": _acc(v["strict"] for v in g.values()),
            "len_mean": round(statistics.mean(n), 2) if n else None, "len_median": statistics.median(n) if n else None,
            "len_at_cap": _acc(x >= max_new for x in n)}


# ------------------------------------------------------------------ stored qa_eval outputs
def stored_outputs(run: Path, qhash: str, file: str | None = None, prefix: str | None = None):
    """{(arm, field, qid): (output, correct)} from the run's qa_eval recall cells (last row wins), or None."""
    p = run / (file or f"outputs-{qhash}.jsonl")
    if not p.exists():
        return None
    prefix = prefix or f"{qhash}."
    out = {}
    for line in p.open():
        r = json.loads(line)
        if r.get("cell", "").startswith(f"{prefix}recall.") and "output" in r:
            out[(r["arm"], r["field"], r["qid"])] = (r["output"], bool(r["correct"]))
    return out


def compare_stored(stored, arm, field, gens):
    if stored is None:
        return {"stored": None}
    sa = STORED_ARM.get(arm, arm)
    both = [q for q in gens if (sa, field, q) in stored]
    diff = sorted(hid(q) for q in both if stored[(sa, field, q)][0] != gens[q]["output"])
    cdiff = [q for q in both if stored[(sa, field, q)][1] != gens[q]["lenient"]]
    return {"stored_arm": sa, "n_compared": len(both), "output_equal": len(both) - len(diff),
            "correct_equal": len(both) - len(cdiff),
            "agreement": round((len(both) - len(diff)) / len(both), 4) if both else None, "differing_qids": diff}


# ------------------------------------------------------------------ contrasts
def did_bootstrap(per_item, docs_of, comp, base="base", n_boot=2000, seed=0):
    """Document-level paired bootstrap of lift(comp) - lift(base) on D_test and held and their difference.
    per_item[(arm, mode)] = {item: 0/1}; docs_of[split] = {doc: [items]}. Documents are resampled with replacement
    within each split; all four conditions use the same draw."""
    conds = [(comp, "realigned"), (comp, "native"), (base, "realigned"), (base, "native")]
    rng = np.random.default_rng(40_000 + seed)
    pts, boots, n = {}, {}, {}
    for split in ("d_test", "held"):
        docs = sorted(docs_of[split])
        S = np.array([[sum(per_item[c][it] for it in docs_of[split][d]) for d in docs] for c in conds], float)
        N = np.array([len(docs_of[split][d]) for d in docs], float)
        n[split] = {"docs": len(docs), "items": int(N.sum())}
        if not len(docs) or N.sum() == 0:
            return {"n": n, "note": "empty split"}

        def stat(s, cnt):
            acc = s / cnt
            return (acc[0] - acc[1]) - (acc[2] - acc[3])
        pts[split] = float(stat(S.sum(1), N.sum()))
        idx = rng.integers(0, len(docs), size=(n_boot, len(docs)))
        boots[split] = stat(S[:, idx].sum(-1), N[idx].sum(-1)[None, :])     # (n_boot,)
    out = {"n": n, "n_boot": n_boot}
    for key, pt, b in (("delta_lift_d_test", pts["d_test"], boots["d_test"]),
                       ("delta_lift_held", pts["held"], boots["held"]),
                       ("did", pts["d_test"] - pts["held"], boots["d_test"] - boots["held"])):
        se = float(np.std(b, ddof=1))
        out[key] = {"est": round(pt, 4), "ci95": [round(float(np.percentile(b, 2.5)), 4),
                                                  round(float(np.percentile(b, 97.5)), 4)],
                    "se": round(se, 4), "mde80": round(2.8 * se, 4)}
    return out


def recall_contrasts(gens, q, comps, eval_fields, a):
    """Per component and scorer: the bootstrap contrasts over both fields pooled and per field."""
    res = {}
    for comp in [c for c, _, _ in comps if c != "base"]:
        for scorer in ("lenient", "strict"):
            for fields in [eval_fields] + ([[f] for f in eval_fields] if len(eval_fields) > 1 else []):
                per_item = {}
                for arm in ("base", comp):
                    for mode in ("native", "realigned"):
                        per_item[(arm, mode)] = {(f, r["qid"]): int(gens[arm][mode][f][s][r["qid"]][scorer])
                                                 for s in ("d_test", "held") for r in q[s] for f in fields}
                docs_of = {s: collections.defaultdict(list) for s in ("d_test", "held")}
                for s in ("d_test", "held"):
                    for r in q[s]:
                        for f in fields:
                            docs_of[s][r["doc_id"]].append((f, r["qid"]))
                key = "+".join(fields)
                res.setdefault(comp, {}).setdefault(scorer, {})[key] = did_bootstrap(
                    per_item, docs_of, comp, n_boot=a.n_boot, seed=a.seed)
    return res


def sep_contrasts(results):
    out = {}
    base = results["base"].get("separability")
    if not base:
        return out
    for arm, r in results.items():
        if arm == "base" or not r.get("separability"):
            continue
        d = {}
        for m in ("ncm_acc", "knn1_acc", "silhouette"):
            ct = {s: (r["separability"][s][m] - base[s][m]) if None not in (r["separability"][s][m], base[s][m])
                  else None for s in ("trained", "held")}
            d[m] = {"component_minus_base_trained": _r(ct["trained"]), "component_minus_base_held": _r(ct["held"]),
                    "trained_minus_held": _r(ct["trained"] - ct["held"]) if None not in ct.values() else None}
        out[arm] = d | {"caveat": SEP_CAVEAT}
    return out


def _r(x):
    return None if x is None else round(x, 4)


# ------------------------------------------------------------------ separability
def sep_chunks(tok, docs, ids, max_docs, max_chunks, seed):
    """{doc id: [chunks]} for documents in ids with >= 2 chunks; at most max_docs documents and max_chunks chunks each
    (seeded selection)."""
    by = collections.defaultdict(list)
    for c in CT.doc_chunks(tok, docs, ids):
        by[c[0]].append(c)
    ok = sorted(k for k, v in by.items() if len(v) >= 2)
    rng = random.Random(30_000 + seed)
    pick = sorted(rng.sample(ok, min(max_docs, len(ok))))
    out = {}
    for k in pick:
        ch = by[k]
        sel = sorted(rng.sample(range(len(ch)), min(max_chunks, len(ch))))
        out[k] = [ch[i] for i in sel]
    return out


def pooled_states(model, fwd, tile, zv, chunks, batch=4):
    """Mean final-norm state over real tokens at positions >= 1: (N, d) float64."""
    tap = Tap(model, "norm")
    out = []
    for i in range(0, len(chunks), batch):
        sub = chunks[i:i + batch]
        T = mx.array([w[:-1] for _, w, _ in sub])
        n_real = [min(n, T.shape[1]) for _, _, n in sub]
        hn, _ = fwd(tap, T, tile, zero_values=zv)
        pos = mx.arange(T.shape[1])[None, :]
        M = ((pos >= 1) & (pos < mx.array(n_real)[:, None])).astype(mx.float32)
        s = mx.sum(hn.astype(mx.float32) * M[..., None], axis=1) / mx.sum(M, axis=1, keepdims=True)
        out.append(np.array(s, dtype=np.float64))
    return np.concatenate(out) if out else np.zeros((0, model.config.hidden_size))


def separability(X, labels):
    """Leave-one-out nearest-class-mean and kNN-1 accuracy of labels, and the silhouette (cosine distance), on
    set-centred, L2-normalised vectors. Every class needs >= 2 members."""
    X = np.asarray(X, dtype=np.float64)
    labels = list(labels)
    cls = sorted(set(labels))
    n = len(labels)
    if len(cls) < 2:
        return {"n_docs": len(cls), "n_chunks": n, "ncm_acc": None, "knn1_acc": None, "silhouette": None}
    Z = X - X.mean(axis=0, keepdims=True)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    y = np.array([cls.index(l) for l in labels])
    counts = np.bincount(y, minlength=len(cls))
    if counts.min() < 2:
        raise ValueError("separability: every class needs >= 2 members")
    sums = np.stack([Z[y == c].sum(axis=0) for c in range(len(cls))])
    ncm = 0
    for i in range(n):
        means = sums / counts[:, None]
        means[y[i]] = (sums[y[i]] - Z[i]) / (counts[y[i]] - 1)
        ncm += int(np.argmin(np.linalg.norm(means - Z[i], axis=1)) == y[i])
    D = np.clip(1.0 - Z @ Z.T, 0.0, 2.0)                           # cosine distance
    Dk = D.copy()
    np.fill_diagonal(Dk, np.inf)
    knn = int(np.sum(y[np.argmin(Dk, axis=1)] == y))
    sil = []
    for i in range(n):
        same = (y == y[i])
        same[i] = False
        a_i = D[i, same].mean()
        b_i = min(D[i, y == c].mean() for c in range(len(cls)) if c != y[i])
        sil.append((b_i - a_i) / max(a_i, b_i, 1e-12))
    return {"n_docs": len(cls), "n_chunks": n, "chance": round(1.0 / len(cls), 4),
            "ncm_acc": round(ncm / n, 4), "knn1_acc": round(knn / n, 4), "silhouette": round(float(np.mean(sil)), 4)}


# ------------------------------------------------------------------ the check
def run_check(W, comps, fwd, rows, a, stored=None, say=print):
    """W: model, tok, docs, train_ids, held_ids. comps: [(arm name, tile or None, zero_values)], base first. rows:
    recall.jsonl rows (file order). stored: stored_outputs() of the run or None. Returns a dict: splits (raw ids),
    q (rows per split), results (per arm), gens (arm -> mode -> field -> split -> qid -> record), contrasts."""
    model, tok = W["model"], W["tok"]
    if comps[0][0] != "base":
        raise ValueError("the first arm must be base")
    splits = doc_split(rows, W["train_ids"], W["held_ids"], a.seed)
    capped = rows_for(rows, splits["d_fit"] + splits["d_test"] + splits["held"])
    if a.max_questions is not None:
        capped = [r for k, s in enumerate(SPLITS) for r in rows_for(rows, splits[s], a.max_questions, a.seed + k)]
        order = {r["qid"]: i for i, r in enumerate(rows)}
        capped.sort(key=lambda r: order[r["qid"]])
    q = {s: [r for r in capped if r["doc_id"] in set(splits[s])] for s in SPLITS}
    trained_rows = [r for r in capped if r["doc_id"] in W["train_ids"]]      # qa_eval's recall.train rows, in order
    held_rows = [r for r in capped if r["doc_id"] in W["held_ids"]]
    train_fields = [f for f in a.train_fields.split(",") if f]
    eval_fields = [f for f in a.eval_fields.split(",") if f]
    ex = answer_examples(tok, q["d_fit"], train_fields)
    if not ex:
        raise SystemExit("no D_fit training questions")
    strict = Strict(tok, rows)
    sep = {}
    if a.sep_docs > 0:
        sep = {"trained": sep_chunks(tok, W["docs"], W["train_ids"], a.sep_docs, a.sep_chunks, a.seed),
               "held": sep_chunks(tok, W["docs"], W["held_ids"], a.sep_docs, a.sep_chunks, a.seed + 1)}
    results, gens = {}, {}
    for arm, tile, zv in comps:
        t0 = time.time()
        res, g = {}, {"native": {}, "realigned": {}}
        for field in eval_fields:                          # native: qa_eval's order and batching (train, then held)
            gt = generate_rows(model, tok, fwd, tile, zv, trained_rows, field, a.max_new, a.gen_batch, strict)
            gh = generate_rows(model, tok, fwd, tile, zv, held_rows, field, a.max_new, a.gen_batch, strict)
            g["native"][field] = {s: {r["qid"]: (gt | gh)[r["qid"]] for r in q[s]} for s in SPLITS}
            res.setdefault("native_vs_stored", {})[field] = compare_stored(stored, arm, field, gt | gh)
        st = answer_states(model, fwd, tile, zv, ex, tok.pad, a.gen_batch)
        ro, log = train_readout(model, st, a)
        res["readout_train"] = {"n_examples": len(ex), "n_target_tokens": int(sum(int(y.size) for _, y in st)),
                                "targets": "answer tokens + <|im_end|>", "log": log}
        say(f"[{arm}] readout trained: loss {log[0]['loss']} -> {log[-1]['loss']}", flush=True)
        tap = Tap(model, "readout", ro)
        for field in eval_fields:
            names = ("d_test", "held") + (("d_fit",) if a.fit_eval else ())
            g["realigned"][field] = {s: generate_rows(tap, tok, fwd, tile, zv, q[s], field, a.max_new, a.gen_batch,
                                                      strict) for s in names}
        for mode in ("native", "realigned"):
            for field in eval_fields:
                for s, gg in g[mode][field].items():
                    res.setdefault(mode, {}).setdefault(field, {})[s] = summary(gg, a.max_new)
        for field in eval_fields:
            nat, rea = res["native"][field], res["realigned"][field]
            res.setdefault("lift", {})[field] = {
                sc: {s: _r(rea[s][sc] - nat[s][sc]) if None not in (rea[s][sc], nat[s][sc]) else None
                     for s in ("d_test", "held")} for sc in ("lenient", "strict")}
        say(f"[{arm}] native {res['native']} realigned {res['realigned']}", flush=True)
        if sep:
            res["separability"] = {}
            for name, by in sep.items():
                ch = [c for k in sorted(by) for c in by[k]]
                X = pooled_states(model, fwd, tile, zv, ch, a.sep_batch)
                res["separability"][name] = separability(X, [c[0] for c in ch])
            res["separability"]["caveat"] = SEP_CAVEAT
            say(f"[{arm}] separability {res['separability']}", flush=True)
        res["seconds"] = round(time.time() - t0, 1)
        results[arm], gens[arm] = res, g
    contrasts = {"recall": recall_contrasts(gens, q, comps, eval_fields, a),
                 "separability": sep_contrasts(results),
                 "answer_overlap_with_d_fit": answer_overlap(q)}
    n = {s: {"docs": len({r["doc_id"] for r in q[s]}), "questions": len(q[s]),
             "items": len(q[s]) * len(eval_fields)} for s in SPLITS}
    return {"splits": splits, "sep": {k: sorted(v) for k, v in sep.items()}, "q": q, "n": n, "results": results,
            "gens": gens, "contrasts": contrasts, "eval_fields": eval_fields, "train_fields": train_fields}


def per_question(gens):
    """Hashed per-question scores: arm -> mode -> field -> split -> qid hash -> {lenient, strict, n_tok}."""
    return {arm: {mode: {f: {s: {hid(qid): {"lenient": v["lenient"], "strict": v["strict"], "n_tok": v["n_tok"]}
                                 for qid, v in d.items()} for s, d in fs.items()} for f, fs in ms.items()}
                  for mode, ms in g.items()} for arm, g in gens.items()}


def output_rows(gens):
    """Private audit rows (generated text): one per arm, mode, field, split, qid."""
    for arm, g in gens.items():
        for mode, ms in g.items():
            for f, fs in ms.items():
                for s, d in fs.items():
                    for qid, v in d.items():
                        yield {"label": LABEL, "arm": arm, "mode": mode, "field": f, "split": s, "qid": qid,
                               "qid_h": hid(qid)} | v


# ------------------------------------------------------------------ loading finished runs
def _side_defaults():
    from research.ens import side_tile as ST
    d = vars(ST.parser().parse_args(["train", "--corpus", "-", "--out", "-"]))
    return {k: v for k, v in d.items() if k not in ("cmd", "corpus", "out")} | \
        {"accum": 1, "clip": 0.0, "track_every": 0}          # side_tile._saved_args: runs before these existed


def load_side_tile(run: Path, corpus=None):
    from research.ens import side_tile as ST
    rec = json.loads((run / "train_receipt.json").read_text())
    targs = argparse.Namespace(**(_side_defaults() | rec["args"]))
    if corpus:
        targs.corpus = corpus
    W = ST.build_world(targs)
    if W["ckpt"] != rec["base_checkpoint"] or W["hashes"] != rec["corpus_hashes"]:
        raise SystemExit("base checkpoint or corpus files differ from the training receipt")
    tile = ST.load_run(run, W, targs)                    # checks the tile fingerprint
    return W, rec, targs, [("base", None, False), ("tile", tile, False)], ST.forward


def load_corpus_tile(run: Path, corpus=None, lora_only_arm=False):
    rec = json.loads((run / "train_receipt.json").read_text())
    targs = argparse.Namespace(**(CT.train_defaults() | rec["args"]))      # older runs predate later arguments
    if corpus:
        targs.corpus = corpus
    W = CT.build_world(targs, run=run)
    if W["ckpt"] != rec["base_checkpoint"] or W["hashes"] != rec["corpus_hashes"]:
        raise SystemExit("base checkpoint or corpus files differ from the training receipt")
    weights = run / "tile.safetensors"
    if CT.sha16(weights)[:12] != rec["tile_checkpoint"].rpartition("@")[2]:
        raise ValueError(f"{weights}: weights differ from the training receipt")
    tile = CT.make_tile(W, targs)
    tile.load_weights(str(weights))
    tile.freeze()
    if getattr(targs, "no_memory", False):
        comps = [("base", None, False), ("lora", tile, False)]
    else:
        comps = [("base", None, False), ("tile", tile, False)]
        if lora_only_arm:
            comps.append(("lora_only", tile, True))
    return W, rec, targs, comps, CT.forward


def stored_qa_batch(run: Path):
    p = run / "receipt.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text()).get("eval", {})
    return ev.get("eval_batches", {}).get("qa_batch", 8 if "qa" in ev else None)


# ------------------------------------------------------------------ receipt
def _rel(run):
    try:
        return os.path.relpath(Path(run).resolve(), C.ROOT)
    except ValueError:
        return Path(run).name


def receipt(run, kind, rec, args, chk, qhash, extra=None):
    name, _, fp = rec["tile_checkpoint"].rpartition("@")
    sp = chk["splits"]
    splits = {k: [hid(x) for x in v] for k, v in sp.items()} | \
        {f"{s}_qids": [hid(r["qid"]) for r in chk["q"][s]] for s in SPLITS} | \
        {f"sep_{k}_docs": [hid(x) for x in v] for k, v in chk["sep"].items()}
    a = vars(args) | {"run": _rel(args.run), "out": Path(args.out).name, "questions": Path(args.questions).name,
                      "corpus": Path(args.corpus).name if args.corpus else None}
    return {"schema": SCHEMA, "label": LABEL, "run_dir": _rel(run), "kind": kind,
            "run_checkpoint": rec["tile_checkpoint"], "run_checkpoint_name": name, "run_checkpoint_fingerprint": fp,
            "base_checkpoint": rec["base_checkpoint"], "corpus_hashes": rec.get("corpus_hashes"),
            "questions_sha12": qhash, "args": a, "n": chk["n"],
            "eval_fields": chk["eval_fields"], "train_fields": chk["train_fields"],
            "readout": {"placement": PLACEMENT, "rank": args.rank, "steps": args.steps, "lr": args.lr,
                        "optimizer": "AdamW, constant lr, weight decay 0", "batch": args.batch, "clip": args.clip,
                        "loss": "answer tokens + <|im_end|>, teacher forced, chat format thinking off",
                        "cached_states": "matches the full forward to allclose 2e-2 (bf16), not bit for bit"},
            "scoring": {"lenient": "corpus_tile_eval.correct over the whole output",
                        "strict": f"answer in the first non-empty line cut to {STRICT_TOKENS} tokens, no other "
                                  "question's answer or alias before it",
                        "lengths": "re-encoded output tokens; len_at_cap = share >= max_new"},
            "contrast_method": "lift = realigned - native; delta_lift = lift(component) - lift(base); did = "
                               "delta_lift(d_test) - delta_lift(held); document-level paired bootstrap within split, "
                               f"{args.n_boot} resamples; mde80 = 2.8 x se",
            "separability_method": "mean final-norm state, positions >= 1; set-centred, L2-normalised; "
                                   "LOO nearest-class-mean, kNN-1, silhouette (cosine)",
            "separability_caveat": SEP_CAVEAT,
            "splits": splits, "arms": chk["results"], "contrasts": chk["contrasts"],
            "per_question": per_question(chk["gens"]), **(extra or {}),
            **C.source_revision(), **C.host_record()}


def outputs_path(out: Path) -> Path:
    return out.with_name(out.stem + ".outputs.jsonl")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--kind", choices=["side_tile", "corpus_tile"], required=True)
    p.add_argument("--questions", required=True, help="eval directory holding recall.jsonl (and crossdoc.jsonl)")
    p.add_argument("--out", required=True, help="receipt JSON path (refused if it or its outputs file exists)")
    p.add_argument("--corpus", default=None, help="override the run's corpus path (same files: hashes are checked)")
    p.add_argument("--lora-only-arm", action="store_true", help="corpus_tile tile runs: add the memory-emptied arm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8, help="questions per readout step")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--train-fields", default="question", help="recall.jsonl fields used as D_fit prompts")
    p.add_argument("--eval-fields", default="question,rephrase", help="fields evaluated")
    p.add_argument("--max-questions", type=int, default=None,
                   help="cap per split (seeded subset; breaks the per-qid check against stored qa_eval outputs)")
    p.add_argument("--no-fit-eval", dest="fit_eval", action="store_false", help="skip realigned recall on D_fit")
    p.add_argument("--max-new", type=int, default=E.MAX_NEW)
    p.add_argument("--gen-batch", type=int, default=8, help="generation batch (qa_eval's default 8)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--sep-docs", type=int, default=32, help="documents per separability set (0 = skip)")
    p.add_argument("--sep-chunks", type=int, default=4, help="chunks per document")
    p.add_argument("--sep-batch", type=int, default=4)
    p.add_argument("--gate-gib", type=float, default=16.0)
    p.add_argument("--no-memory-gate", action="store_true")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    out = Path(a.out)
    for p in (out, outputs_path(out)):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    run = Path(a.run)
    qdir = Path(a.questions).expanduser()
    rows = [json.loads(l) for l in (qdir / "recall.jsonl").open()]
    qhash = questions_hash(qdir)
    if not a.no_memory_gate:
        from research.ens import physics_bus as PB
        PB.memory_gate(gib=a.gate_gib)
    t0 = time.time()
    if a.kind == "side_tile":
        W, rec, _targs, comps, fwd = load_side_tile(run, a.corpus)
    else:
        W, rec, _targs, comps, fwd = load_corpus_tile(run, a.corpus, a.lora_only_arm)
    stored, stored_format = stored_outputs_any(run, qdir)
    print(f"stored qa_eval outputs: {stored_format or 'none'}", flush=True)
    sqb = stored_qa_batch(run)
    if stored is not None and sqb not in (None, a.gen_batch):
        print(f"warning: stored qa_eval used batch {sqb}, this run {a.gen_batch}: outputs may differ", flush=True)
    chk = run_check(W, comps, fwd, rows, a, stored=stored)
    r = receipt(run, a.kind, rec, a, chk, qhash,
                {"stored_qa_eval": {"found": stored is not None, "qa_batch": sqb, "gen_batch": a.gen_batch},
                 "outputs_file": outputs_path(out).name, "seconds": round(time.time() - t0, 1)})
    for p in (out, outputs_path(out)):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p} (appeared during the run)")
    out.parent.mkdir(parents=True, exist_ok=True)
    op = outputs_path(out)
    tmp = op.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        f.write(json.dumps({"schema": OUTPUTS_SCHEMA, "label": LABEL, "receipt": out.name}) + "\n")
        for row in output_rows(chk["gens"]):
            f.write(json.dumps(row) + "\n")
    tmp.replace(op)
    C.write_json(out, r)
    print("wrote", out, "and", op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
