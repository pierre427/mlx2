"""H51: one corpus tile for a whole body of documents (Pierre, 2026-11-24: "Put all of it in a tile. Make sure memory
anchors both topics and concepts. Papers relate to each other, and all papers were stored in the same tile matrix
... test with 0.6b and 4b ... for recall, reasoning over the content"; "also add the contents and lessons from the MCP
tuning library"). Design: wiki/docs/ens/h51-corpus-tile.md.

A memory tile at the bus site of a frozen core (Qwen3 via research/ens/qwen3.py):
  input   x = unit_rms(h) - mu, mu = the mean site state over corpus positions >= 1 (centring makes anchors and
          queries distinguishable; uncentred anchor states had cosine 0.84 with each other, review 2026-11-24)
  query   q = normalise(Wq x), every token position except 0 (position 0 is the attention sink: never read or written)
  keys    anchors: normalise(Wq x_anchor), x_anchor = the frozen core's own centred state for the anchor phrase, read
          after a fixed context ("Topic:") so the phrase is never at the sink; topics (course lectures and the tuning
          library's curated topics), concepts (course phrases) and documents (titles); free keys initialised from real
          query vectors (random free keys never won a read)
  read    top-k slots by cosine x temperature, softmax over the k, weighted sum of learned values (hidden width)
  write   h <- h + bounded(gate * read, h, tau); values start at 0, so the untrained tile is the original model exactly
  port    rank-r LoRA on the blocks after the site, active only when the tile is admitted (H43: the core has to be
          trained to use the bus); the core never changes. --no-memory trains the LoRA alone (the control that says
          whether recall comes from the memory or from the LoRA memorising text)
Loss: next-token NLL on corpus chunks + w_route x routing loss (the log softmax mass, over ALL slots, that a chunk's
tokens put on its document's anchor, its topics and the concepts present in it) + KL to the original model on ordinary
English rows with the tile admitted. Resumable (autosave, SIGTERM), memory gate (--gate-gib) and swap guard.

  python -m research.ens.corpus_tile train --corpus $ENS_PRIVATE_CORPUS/corpus --model Qwen/Qwen3-0.6B --site 8 --out <dir>
  python -m research.ens.corpus_tile eval --run <dir> [--questions $ENS_PRIVATE_CORPUS/eval]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
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
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import lm_bus as LB
from research.ens import physics_bus as PB
from research.ens import qwen3 as Q

RECIPE = "ens.h51.corpus_tile.v2"
CHUNK = 512
ANCHOR_CONTEXT = "Topic:"


def sha16(p: Path) -> str:
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()[:16]


# ------------------------------------------------------------------ corpus, anchors
def load_corpus(d: Path):
    docs = [json.loads(l) for l in open(d / "corpus.jsonl")]
    split = json.loads((d / "split.json").read_text())
    graph = json.loads((d / "graph.json").read_text())
    return docs, split, graph


def topics_of(graph):
    """doc id -> set of topic node names ('term|lecture' for the course, 'tuning|topic' for the library)."""
    out = collections.defaultdict(set)
    for e in graph["edges"]:
        if e["src"].startswith("lecture:") and not e["src"].endswith("|None"):
            out[e["dst"]].add(e["src"][len("lecture:"):])
    return out


def words(text):
    return re.findall(r"[a-z][a-z\-]{1,}", text.lower())   # 2+ letters ("ai", "of": "chain of thought")


def extract_concepts(ref, n_max=1500, min_docs=3, max_group_frac=0.15, group="course", template_group="tuning"):
    """Recurring 1-3 word phrases of the course documents (corpus/ref_df.json, built by
    the corpus build's ref_df.py from the TRAIN split only): in >= min_docs course training documents, in at
    most max_group_frac of either source's documents (more is template text, e.g. the tuning library's record
    labels), and over-represented against FineWeb-Edu: score = course document frequency x log(course doc fraction /
    FineWeb doc fraction), multi-word x1.5, a single word must be 20x over-represented. The tuning library contributes
    its own curated topics as topic anchors instead (its record-structured prose made phrase extraction template)."""
    if n_max <= 0:
        return []
    gd, n_g, n_r = ref["group_df"][group], ref["group_sizes"][group], ref["_n_ref_docs"]
    scored = []
    for g, c in gd.items():
        if c < min_docs:
            continue
        tg = ref["group_df"].get(template_group, {})
        if tg.get(g, 0) / max(1, ref["group_sizes"].get(template_group, 1)) > max_group_frac:
            continue                                     # template text of the tuning library's records (review:
                                                         # capping the course group too dropped its central terms)
        r = ref["ref_df"].get(g, 0)
        ratio = (c / n_g) / ((r + 1) / n_r)
        multi = " " in g
        if ratio < (3.0 if multi else 20.0):
            continue
        scored.append((c * math.log(ratio) * (1.5 if multi else 1.0), g))
    scored.sort(reverse=True)
    out = []
    for _, g in scored:
        if " " not in g and any(g in s.split(" ") for s in out):
            continue                                     # a single word already inside a chosen phrase
        out.append(g)
        if len(out) >= n_max:
            break
    return out


def site_inputs(model, tok, texts, site, n_tok=128, batch=16):
    """unit_rms site states at positions >= 1 of the given texts: (N, d) float32 (for mu and free-key init)."""
    out = []
    for i in range(0, len(texts), batch):
        ids = [tok.encode(t)[:n_tok] for t in texts[i:i + batch]]
        ids = [x for x in ids if len(x) > 8]
        if not ids:
            continue
        L = max(len(x) for x in ids)
        T = mx.array([x + [tok.pad] * (L - len(x)) for x in ids])
        h = PB.unit_rms(PB.prefix(model, T, site))
        for b, x in enumerate(ids):
            out.append(h[b, 1:len(x)])
        mx.eval(out[-1])
    return mx.concatenate(out)


def anchor_states(model, tok, phrases, site, mu, batch=32):
    """Centred frozen-core state for each anchor phrase, read after ANCHOR_CONTEXT so it is never at the sink: mean of
    (unit_rms(h) - mu) over the phrase's tokens, unit RMS: (A, d) float32."""
    ctx = tok.encode(ANCHOR_CONTEXT)
    out = []
    for i in range(0, len(phrases), batch):
        ids = [tok.encode(" " + p)[:24] or [tok.pad] for p in phrases[i:i + batch]]
        L = max(len(x) for x in ids)
        T = mx.array([ctx + x + [tok.pad] * (L - len(x)) for x in ids])
        M = mx.array([[0.0] * len(ctx) + [1.0] * len(x) + [0.0] * (L - len(x)) for x in ids])
        h = PB.unit_rms(PB.prefix(model, T, site)) - mu
        s = mx.sum(h * M[..., None], axis=1) / mx.sum(M, axis=1, keepdims=True)
        out.append(PB.unit_rms(s))
        mx.eval(out[-1])
    return mx.concatenate(out)


# ------------------------------------------------------------------ the tile
class CorpusTile(nn.Module):
    def __init__(self, model, site, H_anchor, mu, n_free=16384, dk=256, k=16, rank=16, tau=0.5, temp=16.0,
                 use_anchors=True, no_memory=False, free_init=None):
        super().__init__()
        c = model.config
        d = c.hidden_size
        self.site, self.k, self.dk, self.tau, self.temp = site, k, dk, tau, temp
        self.no_memory = no_memory
        self.n_anchor = int(H_anchor.shape[0]) if (use_anchors and not no_memory) else 0
        self.n_free = 0 if no_memory else n_free
        self.wq = nn.Linear(d, dk, bias=False)
        if self.n_free:
            if free_init is not None:                        # real query vectors through the initial Wq (review)
                fk = self.wq(free_init[:self.n_free]) + 0.05 * mx.random.normal((self.n_free, dk)) / math.sqrt(dk)
            else:
                fk = mx.random.normal((self.n_free, dk))
            self.free_keys = fk / (mx.linalg.norm(fk, axis=-1, keepdims=True) + 1e-6)
        self.slot_values = mx.zeros((self.n_anchor + self.n_free, d))
        self.gate = mx.array([1.0])
        self._H_anchor = H_anchor if self.n_anchor else None    # fixed, not a parameter (leading underscore)
        self._mu = mu
        if no_memory:
            self.freeze(recurse=False, keys=["slot_values", "gate", "wq"])
        H, K, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        dims = model.lora_dims if hasattr(model, "lora_dims") else (lambda i: {
            "q_proj": (d, H * hd), "k_proj": (d, K * hd), "v_proj": (d, K * hd), "o_proj": (H * hd, d),
            "gate_proj": (d, c.intermediate_size), "up_proj": (d, c.intermediate_size),
            "down_proj": (c.intermediate_size, d)})
        self.lora = {f"b{i}": {n: LB.LoRA(a, b, rank) for n, (a, b) in dims(i).items()}
                     for i in range(site + 1, c.num_hidden_layers)}

    def slot_keys(self):
        parts = []
        if self.n_anchor:
            ka = self.wq(self._H_anchor)
            parts.append(ka / (mx.linalg.norm(ka, axis=-1, keepdims=True) + 1e-6))
        if self.n_free:
            fk = self.free_keys
            parts.append(fk / (mx.linalg.norm(fk, axis=-1, keepdims=True) + 1e-6))
        return mx.concatenate(parts)

    def read(self, h, zero_values=False):
        """h (B, T, d) -> (delta (B, T, d), slot logits (B, T, N) or None). Position 0 reads and writes nothing."""
        if self.no_memory or (self.n_anchor + self.n_free) == 0:
            return mx.zeros(h.shape, dtype=mx.float32), None
        x = PB.unit_rms(h) - self._mu
        q = self.wq(x)
        q = q / (mx.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        s = (q @ self.slot_keys().T) * self.temp
        idx = mx.stop_gradient(mx.argpartition(-s, kth=self.k - 1, axis=-1)[..., :self.k])
        w = mx.softmax(mx.take_along_axis(s, idx, axis=-1).astype(mx.float32), axis=-1)
        values = mx.zeros_like(self.slot_values) if zero_values else self.slot_values
        delta = mx.sum(w[..., None] * values[idx].astype(mx.float32), axis=-2)
        not_sink = (mx.arange(h.shape[1]) > 0).astype(mx.float32)[None, :, None]
        return delta * not_sink, s


def forward(model, tokens, tile=None, zero_values=False):
    """Logits (B, T, V); tile None = the original model; zero_values = the tile's LoRA with an empty memory (eval
    control). Also returns the slot logits at the site (or None)."""
    if tile is None:
        h = model.model.embed_tokens(tokens)
        for blk in model.model.layers:
            h = blk(h)
        return model.head(h), None
    h = PB.prefix(model, tokens, tile.site)
    delta, s = tile.read(h.astype(mx.float32), zero_values=zero_values)
    h = h + LB.bounded(tile.gate * delta, h.astype(mx.float32), tile.tau).astype(h.dtype)
    for i in range(tile.site + 1, len(model.model.layers)):
        h = model.model.layers[i](h, "causal", tile.lora.get(f"b{i}"))
    return model.head(h), s


# ------------------------------------------------------------------ data
def doc_chunks(tok, docs, ids, n_tok=CHUNK):
    """[(doc id, token list of length n_tok + 1, real length)] over the documents in ids (title prepended)."""
    out = []
    for d in docs:
        if d["id"] not in ids:
            continue
        t = tok.encode(f"{d.get('title') or ''}\n\n{d['text']}")
        for j in range(0, max(1, len(t) - 1), n_tok):
            w = t[j:j + n_tok + 1]
            if len(w) >= 64:
                out.append((d["id"], w + [tok.pad] * (n_tok + 1 - len(w)), len(w)))
    return out


def text_chunks(tok, split, n=400, n_tok=128, seed=0):
    """Ordinary English windows (the KL rows): documents with index % 10 == 0 are held out, as lm_bus."""
    rng = random.Random(seed)
    wins = []
    with LB.TEXT.open() as f:
        for i, line in enumerate(f):
            if (i % 10 == 0) != (split == "held"):
                continue
            ids = tok.encode(json.loads(line)["text"])
            wins += [ids[j:j + n_tok] for j in range(0, len(ids) - n_tok + 1, n_tok)]
            if len(wins) >= n * 4:
                break
    rng.shuffle(wins)
    return wins[:n]


def route_targets(chunk_tokens, tok, doc_id, anchor_index, doc_topics, concepts_re):
    """Anchor ids this chunk should read: its document, its topics, the concepts present in it."""
    t = {anchor_index[("doc", doc_id)]} if ("doc", doc_id) in anchor_index else set()
    for top in doc_topics.get(doc_id, ()):
        if ("topic", top) in anchor_index:
            t.add(anchor_index[("topic", top)])
    if concepts_re is not None:
        text = " ".join(words(tok.t.decode(chunk_tokens)))
        for m in concepts_re.finditer(text):
            a = anchor_index.get(("concept", m.group(0)))
            if a is not None:
                t.add(a)
    return sorted(t)


def topic_phrase(t):
    term, name = t.split("|", 1)
    return name.replace("\U0001f517", "").replace("-", " ").strip()


# ------------------------------------------------------------------ world
def build_world(args, run: Path | None = None):
    """Everything both train and eval need. With run (eval): the anchor list, mu and anchor states saved at training
    are loaded and checked instead of recomputed (review: recomputation could shift slots silently)."""
    mx.set_cache_limit(int(PB.CACHE_LIMIT_GIB * 2 ** 30))
    cdir = Path(args.corpus).expanduser()
    docs, split, graph = load_corpus(cdir)
    ref = json.loads((cdir / "ref_df.json").read_text())
    if ref.get("split_sha16") != sha16(cdir / "split.json"):
        raise SystemExit("ref_df.json was built from another split.json: rebuild it (build/ref_df.py)")
    hashes = {n: sha16(cdir / n) for n in ("corpus.jsonl", "split.json", "graph.json", "ref_df.json")}
    path = Q.snapshot_dir(args.model)
    model, cfg, ckpt = Q.load(path, dtype=mx.bfloat16, repo=args.model)
    model.freeze()
    tok = LB.Tok(path)
    train_ids, held_ids = set(split["train"]), set(split["held_out"])
    if train_ids & held_ids:
        raise SystemExit("split.json: a document is both train and held out")
    doc_topics = topics_of(graph)
    topics = sorted({t for doc_id in train_ids for t in doc_topics.get(doc_id, ())})
    concepts = extract_concepts(ref, n_max=args.n_concepts)
    anchors = ([("topic", t, topic_phrase(t)) for t in topics] +
               [("concept", c, c) for c in concepts] +
               [("doc", d["id"], d.get("title") or d["id"]) for d in docs if d["id"] in train_ids])
    # held-out documents get NO anchor and no chunks: the tile must not know they exist (the recall floor)
    if run is not None:
        saved = json.loads((run / "anchors.json").read_text())
        if hashes != saved["hashes"] or (saved.get("base_checkpoint", ckpt) != ckpt):
            raise SystemExit("base checkpoint or corpus files differ from the saved anchors")
        # Older runs included topics unique to held documents. Restore their exact
        # slot ordering when the pinned corpus matches.
        if "base_checkpoint" not in saved:
            all_topics = sorted({t for ts in doc_topics.values() for t in ts})
            anchors = [("topic", t, topic_phrase(t)) for t in all_topics] + [a for a in anchors if a[0] != "topic"]
        if [list(a) for a in anchors] != saved["anchors"]:
            raise SystemExit("anchor list differs from training (corpus files changed?)")
        st = mx.load(str(run / "anchor_state.safetensors"))
        mu, H = st["mu"], st["H"]
        free_init = None
    else:
        sample = [d["text"][i:i + 2000] for d in docs if d["id"] in train_ids
                  for i in range(0, min(len(d["text"]), 20000), 4000)]
        random.Random(5).shuffle(sample)
        X = site_inputs(model, tok, sample[:512], args.site)
        mu = mx.mean(X, axis=0)
        H = anchor_states(model, tok, [p for _, _, p in anchors], args.site, mu)
        perm = mx.array(random.Random(6).sample(range(X.shape[0]), min(X.shape[0], args.n_free)), dtype=mx.int32)
        free_init = X[perm] - mu
        if free_init.shape[0] < args.n_free:                     # tile the sample if it is short
            reps = -(-args.n_free // free_init.shape[0])
            free_init = mx.concatenate([free_init] * reps)[:args.n_free]
    anchor_index = {(k, n): i for i, (k, n, _) in enumerate(anchors)}
    concepts_re = (re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(concepts, key=len, reverse=True)) + r")\b")
                   if concepts else None)
    return dict(docs=docs, split=split, graph=graph, model=model, cfg=cfg, ckpt=ckpt, tok=tok, train_ids=train_ids,
                held_ids=held_ids, doc_topics=doc_topics, anchors=anchors, anchor_index=anchor_index, H=H, mu=mu,
                free_init=free_init, concepts=concepts, concepts_re=concepts_re, hashes=hashes)


def make_tile(W, a):
    return CorpusTile(W["model"], a.site, W["H"], W["mu"], n_free=a.n_free, dk=a.dk, k=a.k, rank=a.rank, tau=a.tau,
                      temp=a.temp, use_anchors=not a.no_anchors, no_memory=a.no_memory, free_init=W["free_init"])


# ------------------------------------------------------------------ train
_FREE = {"max_seconds": None, "no_memory_gate": None, "chunk_tag": None, "gate_gib": None}


def train(args) -> int:
    out = Path(args.out)
    if (out / "train_receipt.json").exists():
        print("already trained", flush=True)
        return 0
    resuming = PB.resume_dir(out) is not None
    STARTUP = {"args.json", "anchor_state.safetensors", "anchors.json", "free_init.safetensors"}
    if not resuming:
        if out.is_dir() and {p.name for p in out.iterdir()} <= STARTUP:
            # a launch that stopped before its first autosave (review): start fresh from the same directory
            for n in STARTUP:
                if (out / n).exists():
                    (out / n).unlink()
        out = C.fresh_out(args.out)
        C.write_json(out / "args.json", vars(args))
    elif json.loads((out / "args.json").read_text()) | _FREE != vars(args) | _FREE:
        raise SystemExit("resume: arguments differ")
    mx.random.seed(args.seed)
    if not args.no_memory_gate:
        PB.memory_gate(gib=args.gate_gib)
    run_for_anchors = out if (out / "anchor_state.safetensors").exists() else None
    W = build_world(args, run=run_for_anchors)
    if run_for_anchors is None:                                  # saved once, so resumes and eval use the same keys
        mx.save_safetensors(str(out / "anchor_state.safetensors"), {"mu": W["mu"], "H": W["H"]})
        C.write_json(out / "anchors.json", {"anchors": [list(a) for a in W["anchors"]], "concepts": W["concepts"],
                                            "context": ANCHOR_CONTEXT, "hashes": W["hashes"], "base_checkpoint": W["ckpt"]})
        if W["free_init"] is not None:
            mx.save_safetensors(str(out / "free_init.safetensors"), {"x": W["free_init"]})
    elif (out / "free_init.safetensors").exists():
        W["free_init"] = mx.load(str(out / "free_init.safetensors"))["x"]
    model, tok = W["model"], W["tok"]
    guard = PB.SwapGuard()
    mx.random.seed(args.seed)
    tile = make_tile(W, args)
    chunks = doc_chunks(tok, W["docs"], W["train_ids"])
    text = text_chunks(tok, "train", n=2000, seed=args.seed)
    rng = random.Random(10_000 + args.seed)
    wu = max(1, min(200, args.steps // 20))
    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, wu), optim.cosine_decay(args.lr, max(1, args.steps - wu), args.lr * 0.1)],
        [wu]), weight_decay=0.0)
    opt.init(tile.trainable_parameters())
    it0, log, t_used = 0, [], 0.0
    if resuming:
        it0, log, t_used = PB.load_state(PB.resume_dir(out), tile, opt, [rng])
        print(f"resumed at {it0}", flush=True)
    A = tile.n_anchor
    use_route = A > 0 and args.w_route > 0
    n_slots = tile.n_anchor + tile.n_free

    def lf(t, T, Y, Wm, R, ttok, tref):
        logits, s = forward(model, T, t)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), Y, reduction="none")
        nll = mx.sum(ce * Wm) / mx.maximum(mx.sum(Wm), 1.0)
        route = mx.array(0.0)
        if use_route:
            # log of the softmax mass over ALL slots that falls on the chunk's target anchors (positions >= 1)
            # free keys get no routing gradient (review: a softmax over all slots pushed every free key away from
            # every query, killing free slots during training); they still compete in the denominator
            s32 = mx.concatenate([s[..., :A], mx.stop_gradient(s[..., A:])], axis=-1).astype(mx.float32)
            lp = s32 - mx.logsumexp(s32, axis=-1, keepdims=True)
            tgt = mx.logsumexp(mx.where(R[:, None, :], lp[..., :A], -1e9), axis=-1)
            Wr = Wm * (mx.arange(T.shape[1]) > 0)[None, :]
            route = -mx.sum(tgt * Wr) / mx.maximum(mx.sum(Wr), 1.0)
        tl, _ = forward(model, ttok, t)
        tl = tl.astype(mx.float32)
        kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1)), -1))
        return nll + args.w_route * route + args.kl_weight * kl, (nll, route, kl, s)
    vg = nn.value_and_grad(tile, lambda t, *a: lf(t, *a)[0])
    n_text = max(1, int(round(args.batch * args.p_text)))
    PB._TERM["requested"] = False
    old = signal.signal(signal.SIGTERM, PB._on_term)
    t0, it = time.time(), it0
    try:
        while it < args.steps:
            n_launch = it - it0
            swap = None
            if not args.no_memory_gate and (n_launch <= PB.SWAP_WARMUP_STEPS or n_launch % 10 == 0):
                swap = guard.check(n_launch)
            if PB._TERM["requested"] or swap or (args.max_seconds and time.time() - t0 > args.max_seconds):
                if swap:
                    log.append({"step": it, "swap_stop": swap})
                PB.save_state(out, tile.parameters(), opt, it, [rng], log, t_used + time.time() - t0)
                print(f"stopped at {it} (resume state saved){' - ' + swap if swap else ''}", flush=True)
                return 3
            it += 1
            pick = [chunks[rng.randrange(len(chunks))] for _ in range(args.batch - n_text)]
            T = mx.array([w[:-1] for _, w, _ in pick])
            Y = mx.array([w[1:] for _, w, _ in pick])
            Wm = mx.array([[1.0] * (n - 1) + [0.0] * (CHUNK - n + 1) for _, _, n in pick])
            if use_route:
                Rl = [set(route_targets(w[:n], tok, did, W["anchor_index"], W["doc_topics"], W["concepts_re"]))
                      for did, w, n in pick]
                R = mx.array([[a in r for a in range(A)] for r in Rl])
            else:
                R = mx.zeros((len(pick), 1), dtype=mx.bool_)
            ttok = mx.array([text[rng.randrange(len(text))] for _ in range(n_text)])
            tref = mx.stop_gradient(forward(model, ttok, None)[0].astype(mx.float32))
            if it % args.log_every == 0 or it == args.steps:
                _, (nll, route, kl, s) = lf(tile, T, Y, Wm, R, ttok, tref)       # before the update (review)
                diag = {}
                if s is not None:
                    top = mx.argmax(s[:, 1:], axis=-1)
                    real = Wm[:, 1:] > 0
                    tl = [x for x, r in zip(top.reshape(-1).tolist(), real.reshape(-1).tolist()) if r]
                    diag = {"top1_anchor_frac": round(sum(x < A for x in tl) / max(1, len(tl)), 3),
                            "slots_hit": len(set(tl)), "slots_hit_ceiling": min(len(tl), n_slots)}
            loss, grads = vg(tile, T, Y, Wm, R, ttok, tref)
            opt.update(tile, grads)
            mx.eval(tile.parameters(), opt.state)
            if it % args.log_every == 0 or it == args.steps:
                log.append({"step": it, "loss": round(float(loss.item()), 5), "nll": round(float(nll.item()), 4),
                            "route": round(float(route.item()), 4), "kl": round(float(kl.item()), 5), **diag,
                            "elapsed_s": round(t_used + time.time() - t0, 1), **PB.mem_stats()})
                print(log[-1], flush=True)
            if args.save_every and it % args.save_every == 0 and it < args.steps:
                PB.save_state(out, tile.parameters(), opt, it, [rng], log, t_used + time.time() - t0)
        PB.save_state(out, tile.parameters(), opt, it, [rng], log, t_used + time.time() - t0)
    finally:
        signal.signal(signal.SIGTERM, old)
    wpath = out / "tile.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(tile.parameters())))
    tag = "-lora-only" if args.no_memory else ("-noanchor" if args.no_anchors else "")
    receipt = {"schema": "ens.h51.corpus_tile_train.v2", "recipe": RECIPE, "base_checkpoint": W["ckpt"],
               "tile_checkpoint": f"ens-h51-corpustile-{args.model.split('/')[-1].lower()}-site{args.site}{tag}"
                                  f"-s{args.seed}@{sha16(wpath)[:12]}",
               "corpus_hashes": W["hashes"], "label": "private_corpus: never publish",
               "n_anchor": tile.n_anchor, "n_free": tile.n_free,
               "n_topics": sum(1 for a in W["anchors"] if a[0] == "topic"), "n_concepts": len(W["concepts"]),
               "n_doc_anchors": sum(1 for a in W["anchors"] if a[0] == "doc"), "n_train_chunks": len(chunks),
               "epochs": round(args.steps * (args.batch - n_text) / max(1, len(chunks)), 2),
               "trainable_params": C.n_params(tile), "args": vars(args), "log": log,
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "train_receipt.json", receipt)
    print("trained", receipt["tile_checkpoint"], flush=True)
    return 0


# ------------------------------------------------------------------ evaluation (language-model recall; QA in corpus_tile_eval)
ARMS = (("base", None, False), ("tile", True, False), ("lora_only", True, True))


def nll_eval(W, tile, args, n=64):
    """Mean next-token NLL (positions >= 1) on chunks of trained and never-trained documents and on held ordinary text:
    base, tile, and the tile with its memory emptied (lora_only: the LoRA's share of any gain)."""
    model, tok = W["model"], W["tok"]
    res = {}
    rng = random.Random(70_000 + args.seed)
    is_t = {d["id"]: d["batch"] == "tuning" for d in W["docs"]}
    sets = {"train_docs": doc_chunks(tok, W["docs"], W["train_ids"]),
            "held_docs": doc_chunks(tok, W["docs"], W["held_ids"]),
            "train_course": doc_chunks(tok, W["docs"], {i for i in W["train_ids"] if not is_t[i]}),
            "held_course": doc_chunks(tok, W["docs"], {i for i in W["held_ids"] if not is_t[i]}),
            "train_tuning": doc_chunks(tok, W["docs"], {i for i in W["train_ids"] if is_t[i]}),
            "held_tuning": doc_chunks(tok, W["docs"], {i for i in W["held_ids"] if is_t[i]})}
    for name, ch in sets.items():
        if not ch:
            res[name] = {"n_tokens": 0}
            continue
        pick = rng.sample(ch, min(n, len(ch)))
        acc, cnt = collections.Counter(), 0.0
        for i in range(0, len(pick), 4):
            sub = pick[i:i + 4]
            T = mx.array([w[:-1] for _, w, _ in sub]); Y = mx.array([w[1:] for _, w, _ in sub])
            Wm = mx.array([[1.0] * (m - 1) + [0.0] * (CHUNK - m + 1) for _, _, m in sub])
            for arm, t, zv in ARMS:
                lg, _ = forward(model, T, tile if t else None, zero_values=zv)
                ce = nn.losses.cross_entropy(lg.astype(mx.float32), Y, reduction="none")
                acc[arm] += float(mx.sum(ce * Wm).item())
            cnt += float(mx.sum(Wm).item())
        r = {a: round(acc[a] / cnt, 4) for a, _, _ in ARMS}
        r["gain"] = round(r["base"] - r["tile"], 4)
        r["gain_lora_only"] = round(r["base"] - r["lora_only"], 4)
        res[name] = dict(r, n_tokens=int(cnt))
    text = text_chunks(tok, "held", n=128, seed=1)
    acc = collections.Counter()
    k = 0
    for i in range(0, len(text), 16):
        T = mx.array(text[i:i + 16])
        k += T[:, 1:].size
        for arm, t, zv in ARMS:
            lg, _ = forward(model, T, tile if t else None, zero_values=zv)
            acc[arm] += float(nn.losses.cross_entropy(lg[:, :-1].astype(mx.float32), T[:, 1:], reduction="sum").item())
    if not k:
        raise ValueError("ordinary-text evaluation needs at least one complete held window")
    res["ordinary_text"] = {a: round(acc[a] / k, 4) for a, _, _ in ARMS} | {"cost": round((acc["tile"] - acc["base"]) / k, 4)}
    for src, metric in (("docs", "recall_gain_trained_minus_held"), ("course", "recall_gain_course"),
                        ("tuning", "recall_gain_tuning")):
        train, held = res[f"train_{src}"], res[f"held_{src}"]
        res[metric] = round(train["gain"] - held["gain"], 4) if train["n_tokens"] and held["n_tokens"] else None
    train = res["train_docs"]
    res["memory_share_trained"] = round(train["gain"] - train["gain_lora_only"], 4) if train["n_tokens"] else None
    return res


def evaluate(args) -> int:
    run = Path(args.run)
    rec = json.loads((run / "train_receipt.json").read_text())
    targs = argparse.Namespace(**rec["args"])
    if not args.no_memory_gate:
        PB.memory_gate(gib=args.gate_gib)
    W = build_world(targs, run=run)
    if W["ckpt"] != rec["base_checkpoint"] or W["hashes"] != rec["corpus_hashes"]:
        raise SystemExit("base checkpoint or corpus files differ from the training receipt")
    tile = make_tile(W, targs)
    C.load_checkpoint_weights(tile, run / "tile.safetensors", rec["tile_checkpoint"])
    tile.freeze()
    res = {"nll": nll_eval(W, tile, targs)}
    if args.questions:
        from research.ens import corpus_tile_eval as E
        res |= E.qa_eval(W, tile, targs, Path(args.questions).expanduser(), run)
    receipt = rec | {"schema": "ens.h51.corpus_tile_eval.v2", "eval": res,
                     "eval_source_revision": C.source_revision(), "eval_host": C.host_record()}
    C.write_json(run / "receipt.json", receipt)
    print("eval", json.dumps(res), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--corpus", required=True)
    t.add_argument("--model", default=Q.QWEN3_06B)
    t.add_argument("--site", type=int, default=8)
    t.add_argument("--n-free", type=int, default=16384)
    t.add_argument("--n-concepts", type=int, default=1500)
    t.add_argument("--dk", type=int, default=256)
    t.add_argument("--k", type=int, default=16)
    t.add_argument("--temp", type=float, default=16.0, help="softmax temperature on cosine scores")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--tau", type=float, default=0.5)
    t.add_argument("--no-anchors", action="store_true", help="ablation: free slots only, no routing loss")
    t.add_argument("--no-memory", action="store_true", help="control: the LoRA alone, no memory slots")
    t.add_argument("--w-route", type=float, default=0.1)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--p-text", type=float, default=0.25)
    t.add_argument("--kl-weight", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--max-seconds", type=float, default=None)
    t.add_argument("--gate-gib", type=float, default=24.0, help="free memory required before loading")
    t.add_argument("--no-memory-gate", action="store_true")
    t.add_argument("--chunk-tag", default=None)
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--questions", default=None)
    e.add_argument("--gate-gib", type=float, default=24.0)
    e.add_argument("--no-memory-gate", action="store_true")
    a = p.parse_args(argv)
    sys.exit(train(a) if a.cmd == "train" else evaluate(a))


if __name__ == "__main__":
    main()
