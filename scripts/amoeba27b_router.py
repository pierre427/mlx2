#!/usr/bin/env python3
"""Router v0 for the full-partition tiles (FULL_PARTITION_DESIGN.md): per request, no fitted gates in the model.

CPU only. Features: the mean of the prompt's input embeddings (the 27B's own embed_tokens rows, dequantized),
available before prefill. One-vs-rest logistic probes per router topic:
- the v4 bucket families (enriched atlas corpus cells -> family, via amoeba27b_buckets.FAMILIES);
- the named task boxes (task prompts from amoeba27b_lib.build_tasks).
Fit on a source-hashed fit half, threshold per topic on the held half at --min-recall (the receipt's
threshold_set_* numbers are measured on those same rows; cross_* are two-fold out-of-sample estimates of the same
threshold rule inside the held half). At serve time a request
opens every topic whose probe passes its threshold. If none passes, it opens every tile (the full model), so
a router miss costs speed, never accuracy. Tiles open on the OR of their topics (partition.json).
Also reports, for diagnosis only, which tiles each 20x20 sanity task would open (never used to fit or choose).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amoeba27b_buckets as BK  # noqa: E402

# task prompt -> router topics (the named boxes and the bucket families that hold them)
TASK_TOPICS = {
    "capitals": ["box:capitals", "geography_facts"],
    "capitals_phrasing": ["box:capitals", "geography_facts"],
    "arithmetic": ["box:arithmetic", "box:refined_pool_C32", "math"],
    "multiplication": ["box:arithmetic", "box:refined_pool_C32", "math"],
    "french": ["box:french", "box:refined_pool_C60", "languages"],
    "recall": ["recall"],
    "logic_entail": ["entailment"],
}


# v1 (2026-11-15): chat-style prompts per topic, so the router sees the distribution it serves. Items are
# disjoint from the 20x20 sanity tasks (other countries, other words, other numbers); the task *types*
# overlap by design (a router must recognise "a capital question"), which is recorded with the results.
SANITY_COUNTRIES = {"France", "Japan", "Italy", "Egypt", "Canada", "Spain", "Germany", "Kenya", "Peru", "Norway", "Greece",
                    "Portugal", "Austria", "Ireland", "Cuba", "Poland", "Sweden", "Finland", "Hungary", "Chile"}
FR_FIT_WORDS = ["tree", "window", "bread", "river", "chair", "apple", "school", "friend", "garden", "moon", "table",
                "flower", "street", "train", "mountain", "cheese", "milk", "city", "summer", "winter", "door", "bird",
                "horse", "boat", "shirt", "letter", "kitchen", "forest", "island", "teacher", "child", "night"]
BROAD = {"narrative", "textbook", "howto", "science", "dialogue", "encyclopedic"}


def chat_rows(seed=0):
    import random
    import amoeba27b_lib as A
    rng = random.Random(seed)
    rows = []
    caps = [(c, k) for c, k in A.COUNTRIES if c not in SANITY_COUNTRIES]
    cap_t = ["What is the capital of {}?", "Which city is the capital of {}? Answer briefly.", "Name the capital city of {}.",
             "Tell me two facts about the capital of {}.", "Is the capital of {} on the coast? Name it first."]
    for i, (c, _) in enumerate(caps):
        rows.append((cap_t[i % len(cap_t)].format(c), ["box:capitals", "geography_facts"], f"chat:cap:{c}"))
    ar_t = ["What is {} + {}?", "Add {} and {}.", "Compute {} minus {}.", "What is {} times {}?", "Multiply {} by {}.",
            "What is {} divided by {}? Give the quotient.", "Calculate {} + {} and explain briefly."]
    for i in range(240):
        a, b = rng.randint(2, 999), rng.randint(2, 99)
        rows.append((ar_t[i % len(ar_t)].format(a, b), ["box:arithmetic", "box:refined_pool_C32", "math"], f"chat:ar:{i}"))
    try:
        import pyarrow.parquet as pq   # GSM8K *train* only (never test)
        t = pq.read_table(str(Path.home() / "corpora/benchmarks/openai__gsm8k@740312ad/main/train-00000-of-00001.parquet")).to_pylist()
        for i, r in enumerate(t[:200]):
            rows.append((r["question"], ["math"], f"chat:gsm8k-train:{i}"))
    except Exception:  # noqa: BLE001 - the main venv has no pyarrow; the arithmetic prompts cover math
        pass
    fr_t = ["Translate the word '{}' into French.", "How do you say '{}' in French?", "What is the French for '{}'? Use it in a sentence.",
            "Give the French translation of '{}'."]
    for i, w in enumerate(FR_FIT_WORDS):
        for j in range(2):
            rows.append((fr_t[(i + j) % len(fr_t)].format(w), ["box:french", "box:refined_pool_C60", "languages"], f"chat:fr:{w}:{j}"))
    for i in range(60):
        n = rng.randint(10, 999)
        rows.append((rng.choice(["Remember the number {}. ", "Please keep {} in mind. "]).format(n)
                     + rng.choice(["What number did I give you?", "Repeat the number I told you."]), ["recall"], f"chat:recall:{i}"))
    return rows


class Embedder:
    """Mean input embedding of a text, from the model's own quantized embed_tokens rows.

    MLX loads the shard once; the packed weights, scales and biases are kept as numpy arrays and rows are
    dequantized in numpy, so __call__ is thread-safe (the 20x20 harness routes from worker threads, where an
    MLX CPU stream does not exist; gS2 2026-11-15)."""

    def __init__(self, model_dir):
        import mlx.core as mx
        from transformers import AutoTokenizer
        idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        key = next(k for k in idx if k.endswith("embed_tokens.weight"))
        w = mx.load(str(Path(model_dir) / idx[key]))
        stem = key[:-len(".weight")]
        cfg = json.loads((Path(model_dir) / "config.json").read_text())["quantization"]
        self.gs, self.bits = int(cfg["group_size"]), int(cfg["bits"])
        if cfg.get("mode", "affine") != "affine" or self.bits != 8:
            raise ValueError("router embedder supports 8-bit affine embed_tokens only")
        self.w = np.array(w[stem + ".weight"]).view(np.uint8)          # [V, d] (4 uint8 per packed uint32)
        self.s = np.array(w[stem + ".scales"].astype(mx.float32))       # [V, d / gs]
        self.b = np.array(w[stem + ".biases"].astype(mx.float32))
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))

    def __call__(self, text, max_tokens=512):
        ids = self.tok.encode(text, add_special_tokens=False)[:max_tokens] or [0]
        q = self.w[ids].astype(np.float32)                              # [n, d]
        n, d = q.shape
        q = q.reshape(n, d // self.gs, self.gs) * self.s[ids][..., None] + self.b[ids][..., None]
        return q.reshape(n, d).mean(0)


def half_of(src: str) -> int:
    return int(hashlib.sha256(src.encode()).hexdigest()[:8], 16) % 2


def fold_of(src: str) -> int:
    """A second source hash (independent of half_of's bits) that splits the held half in two."""
    return int(hashlib.sha256(src.encode()).hexdigest()[8:16], 16) % 2


def threshold_report(pos, neg, pos_fold, neg_fold, min_recall):
    """(threshold, report) for one topic from its held-half scores. The threshold is the (1 - min_recall) quantile
    of the positives, so recall on those rows is >= min_recall by construction: those numbers are named
    threshold_set_* (in-sample). cross_recall / cross_fpr apply the same rule two-fold: the threshold from one
    fold of the held rows, measured on the other fold, pooled over both directions (None without positives in
    both folds); they estimate the rule out of sample, not the shipped threshold itself."""
    pos, neg, pos_fold, neg_fold = (np.asarray(x) for x in (pos, neg, pos_fold, neg_fold))
    thr = float(np.quantile(pos, 1 - min_recall)) if len(pos) else 0.0
    rep = {"held_pos": int(len(pos)), "threshold_from": "held_positives" if len(pos) else "none (tau 0.0)",
           "threshold_set_recall": float((pos >= thr).mean()) if len(pos) else None,
           "threshold_set_fpr": float((neg >= thr).mean()) if len(neg) else None,
           "cross_recall": None, "cross_fpr": None}
    if all((pos_fold == f).any() for f in (0, 1)):
        hit = n_pos = fire = n_neg = 0
        for f in (0, 1):
            t = float(np.quantile(pos[pos_fold == f], 1 - min_recall))
            other_pos, other_neg = pos[pos_fold != f], neg[neg_fold != f]
            hit, n_pos = hit + int((other_pos >= t).sum()), n_pos + len(other_pos)
            fire, n_neg = fire + int((other_neg >= t).sum()), n_neg + len(other_neg)
        rep["cross_recall"] = hit / n_pos
        rep["cross_fpr"] = fire / n_neg if n_neg else None
    return thr, rep


def fit_probe(X, y, steps=400, lr=0.5, l2=1e-3):
    w, b = np.zeros(X.shape[1]), 0.0
    pw = (1 - y.mean()) / max(y.mean(), 1e-6)
    sw = np.where(y > 0, pw, 1.0)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2"))
    ap.add_argument("--partition", required=True, help="partition dir (partition.json)")
    ap.add_argument("--enriched", default=str(Path.home() / "corpora/mixed/atlas-enriched-v1/enriched.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-recall", type=float, default=0.98)
    ap.add_argument("--max-texts-per-family", type=int, default=400)
    ap.add_argument("--chat", action="store_true", help="v1: add chat-style prompts per topic; broad registers alone open all")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    emb = Embedder(a.model)
    rows = []                                             # (text, topics, source)
    per = {}
    for line in open(a.enriched):
        r = json.loads(line)
        c = BK.canon(r["cell"])
        fam = next((f for f, pats in BK.FAMILIES.items() if any(fnmatch.fnmatch(c, p) for p in pats)), None)
        if fam and not fam.startswith("task") and per.get(fam, 0) < a.max_texts_per_family:
            per[fam] = per.get(fam, 0) + 1
            rows.append((r["text"], [fam], r.get("source", "") + str(len(rows))))
    import amoeba27b_lib as A

    class _H:                                             # build_tasks needs only encode()
        encode = staticmethod(lambda s: emb.tok.encode(s, add_special_tokens=False))
    tasks, _ = A.build_tasks(_H, seed=0)
    for name, items in tasks.items():
        for i, it in enumerate(items):
            if name in TASK_TOPICS:
                rows.append((it["prompt"], TASK_TOPICS[name], f"task:{name}:{i}"))
    if a.chat:
        rows += chat_rows()
    topics = sorted({t for _, ts, _ in rows for t in ts})
    X = np.stack([emb(t) for t, _, _ in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    half = np.array([half_of(s) for _, _, s in rows])
    fit, held = np.where(half == 0)[0], np.where(half == 1)[0]
    fold = np.array([fold_of(s) for _, _, s in rows])[held]
    W, B, tau, report = [], [], [], {}
    for t in topics:
        y = np.array([1.0 if t in ts else 0.0 for _, ts, _ in rows])
        w, b = fit_probe(Z[fit], y[fit])
        s = Z[held] @ w + b
        yp = y[held] > 0
        # the shipped threshold is unchanged (all held positives); the report says which numbers are in-sample
        thr, report[t] = threshold_report(s[yp], s[~yp], fold[yp], fold[~yp], a.min_recall)
        W.append(w); B.append(b); tau.append(thr)
    W, B, tau = np.stack(W), np.array(B), np.array(tau)
    part = json.loads((Path(a.partition) / "partition.json").read_text())
    fam_of_bucket = {}
    binfo = json.loads((Path(part["buckets"]) / "buckets.json").read_text()) if Path(part["buckets"]).exists() else {"buckets": {}}
    for bname in binfo.get("buckets", {}):
        fam_of_bucket[bname] = bname                      # v4 buckets are named by family
    tile_topics = {}
    for r in part["tiles"]:
        tile_topics[r["tile"]] = sorted(set([f"box:{x}" for x in r["boxes"]] + [fam_of_bucket.get(x, x) for x in r["opens_on_buckets"]]))

    def route(text):
        z = (emb(text) - mu) / sd
        on = [topics[i] for i in np.where(z @ W.T + B >= tau)[0]]
        if not on or (a.chat and set(on) <= BROAD):
            return None, [t["tile"] for t in part["tiles"]]       # nothing specific: open everything
        tiles = [tid for tid, tt in tile_topics.items() if set(tt) & set(on)]
        return on, tiles
    # diagnostic only: what the 20x20 sanity prompts would open (never used to fit or choose thresholds)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sl", Path(__file__).resolve().parent / "amoeba_sanity_lanes.py")
    diag = {}
    try:
        sys.argv = ["x", "http://127.0.0.1:1", "/dev/null"]
        import types
        src = (Path(__file__).resolve().parent / "amoeba_sanity_lanes.py").read_text()
        src = src.replace("THINK = _server_thinks_by_default()", "THINK = False")
        mod = types.ModuleType("sl")
        exec(compile(src.split("\ndef main():")[0], "sl", "exec"), mod.__dict__)
        for name, body, _ in mod.tasks(0):
            text = "\n".join(m.get("content", "") for m in body.get("messages", []) if m.get("role") == "user")
            on, tiles = route(text)
            share = sum(next(r["neurons"] for r in part["tiles"] if r["tile"] == t) for t in tiles) / 1114112
            diag[name] = {"topics": on, "open_all": on is None, "tiles_open": len(tiles), "open_neuron_share": round(share, 4)}
    except Exception as error:                            # noqa: BLE001 - a diagnostic must not fail the fit
        diag = {"error": repr(error)}
    np.savez_compressed(out / "router.npz", W=W.astype(np.float32), B=B.astype(np.float32), tau=tau.astype(np.float32),
                        mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    summary = {"schema": "mlx2.amoeba-router.v0", "features": "mean input embedding of the user text (embed_tokens, dequantized)",
               "model": a.model, "partition": a.partition, "topics": topics, "tile_topics": tile_topics,
               "min_recall": a.min_recall, "n_texts": len(rows), "fit": int(len(fit)), "held": int(len(held)),
               "per_topic_held": report, "fallback": "no topic passes -> open every tile (the full model)"
               + ("; only broad registers pass -> open every tile" if a.chat else ""), "version": "v1-chat" if a.chat else "v0",
               "sanity_20x20_diagnostic_only": diag}
    (out / "router.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"topics": len(topics), "held": report, "sanity": diag}, indent=1)[:6000])


if __name__ == "__main__":
    main()


def safe_tile_id(tile):
    """Runtime tile ids must match [A-Za-z0-9_-]{1,96}: taxonomy tile ids keep their full name in the manifest's
    "name" field and are served under this id (model prefix dropped, '.' and '+' -> '_')."""
    import re
    t = tile.split(".tile.", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_-]", "_", t)[:96]


class Router:
    """Serve-time router v0/v1 from a fitted dir (router.npz + router.json): route(text) -> (topics | None,
    open tile ids). None topics = nothing specific passed (v1: or only broad registers) -> every tile opens."""

    def __init__(self, router_dir, model=None):
        d = Path(router_dir)
        self.info = json.loads((d / "router.json").read_text())
        z = np.load(d / "router.npz")
        self.W, self.B, self.tau, self.mu, self.sd = z["W"], z["B"], z["tau"], z["mu"], z["sd"]
        self.topics = self.info["topics"]
        self.tile_topics = self.info["tile_topics"]
        self.v1 = self.info.get("version") == "v1-chat"
        self.emb = Embedder(model or self.info["model"])

    def route(self, text):
        z = (self.emb(text) - self.mu) / self.sd
        on = [self.topics[i] for i in np.where(z @ self.W.T + self.B >= self.tau)[0]]
        if not on or (self.v1 and set(on) <= BROAD):
            return None, sorted(safe_tile_id(t) for t in self.tile_topics)
        return on, sorted(safe_tile_id(t) for t, tt in self.tile_topics.items() if set(tt) & set(on))
