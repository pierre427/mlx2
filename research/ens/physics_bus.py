"""ENS H43 on a real LM, step 1: a record bus on frozen Qwen3-4B for retrieval and reasoning over 2026 physics papers.

Pierre (2026-11-23): "try it on the m5 with a qwen 4b as the core, and try to add knowledge to it like we did earlier
today on 0.6b", "try using the advanced physics corpus", "just do advanced topics from what we have in the corpus, not
just fact retrieval but also reasoning over it", "if you need to grab the 16bit version go for it".

Knowledge: physics_facts.py (2026 arXiv theory-physics papers; facts per paper: field, posting month, a physics
concept from its abstract). Papers postdate Qwen3, so the base model cannot know them.

The bus (exists only on the bus path, as in lm_bus / lm_typed; with no record the model is the original bit for bit):
at each paper **mention** (the position of the closing quote after its title) a **record** is written: three slots
(field, month, concept), each a payload embedded with the model's own frozen token embedding (or, for a learned
tile, a soft mixture of the slot's value embeddings), projected per slot, summed with a slot-presence op embedding,
bounded to tau * ||h|| and added after block `site`. A LoRA rides on the blocks after the site. Up to two mentions
per prompt.

Tasks: realistic chat Q&A with full-sentence answers (physics_qa.py; Pierre: no single-token replies):
  retrieval  one paper's field, posting month or a central concept
  reasoning  two papers: same field? which came out first? which one deals with <concept>? (decision + facts)
Training is teacher-forced over the whole answer; evaluation generates freely and parses the answer.
Train and held question sets per task.

This module trains the adapter once with a mock provider: titles from harvest papers **outside** the knowledge world,
with random record values, so it learns to read records (retrieval) and to compare two records (reasoning), and
cannot memorise. Resumable (autosave, TERM-safe) for 20-minute M5 slots.

  python -m research.ens.physics_bus --facts <facts.json> --site 10 --steps 3000 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from research.ens import common as C
from research.ens import lm_bus as LB
from research.ens import physics_facts as PF
from research.ens import physics_qa as QA
from research.ens import qwen3 as Q

SEQ = 160
SLOTS = ("field", "month", "concept")
RECIPE = "ens.h43.physics_bus.v2"          # v2: realistic chat Q&A, full-sentence answers (Pierre)

T_TRAIN, T_HELD = QA.Q_TRAIN, QA.Q_HELD        # realistic chat questions, full-sentence answers (physics_qa)
RETRIEVAL, REASONING = ("field", "month", "concept"), ("same_field", "earlier", "which_concept")


# ------------------------------------------------------------------ vocab of slot values
class Vocab:
    def __init__(self, tok, facts, model=None):
        """model with set_virtual (Kimi backend): a multi-token value gets a virtual id whose embedding is the mean of
        its tokens' embeddings. Without it (Qwen3) every value must be one token, as before."""
        fields = sorted(set(PF.FIELDS.values()))
        months = PF.MONTHS
        concepts = sorted({p["concept"] for p in facts["papers"]})
        self.values = {"field": fields, "month": months, "concept": concepts}
        virtual = []
        self.ids = {}
        for s, vs in self.values.items():
            self.ids[s] = []
            for v in vs:
                t = tok.encode(" " + v)
                if len(t) == 1:
                    self.ids[s].append(t[0])
                elif model is not None and hasattr(model, "set_virtual"):
                    self.ids[s].append(model.vocab_size + len(virtual))
                    virtual.append(t)
                else:
                    raise AssertionError(f"{s} value {v!r} is not a single token")
        self.virtual = virtual
        if virtual:
            model.set_virtual(virtual)
        self.yes, self.no = tok.encode(" Yes")[0], tok.encode(" No")[0]
        self.first, self.second = tok.encode(" first")[0], tok.encode(" second")[0]


# ------------------------------------------------------------------ record port
class RecordPort(nn.Module):
    def __init__(self, model: Q.Qwen3, site: int, rank: int = 16, tau: float = 0.5):
        super().__init__()
        c = model.config
        d = c.hidden_size
        self.site, self.tau, self.rank = site, tau, rank
        self.slot_op = nn.Embedding(len(SLOTS), d)
        self.slot_op.weight = self.slot_op.weight * 0.01
        self.proj = {s: nn.Linear(d, d, bias=False) for s in SLOTS}
        for s in SLOTS:
            self.proj[s].weight = mx.random.normal((d, d)) * (0.01 / d ** 0.5)
        self.gate = mx.array([1.0])
        if hasattr(model, "lora_dims"):                 # Kimi backend: per-layer projection shapes (KDA vs MLA)
            self.lora = {f"b{i}": {n: LB.LoRA(a, b, rank) for n, (a, b) in model.lora_dims(i).items()}
                         for i in range(site + 1, c.num_hidden_layers)}
        else:
            H, K, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
            dims = {"q_proj": (d, H * hd), "k_proj": (d, K * hd), "v_proj": (d, K * hd), "o_proj": (H * hd, d),
                    "gate_proj": (d, c.intermediate_size), "up_proj": (d, c.intermediate_size),
                    "down_proj": (c.intermediate_size, d)}
            self.lora = {f"b{i}": {n: LB.LoRA(a, b, rank) for n, (a, b) in dims.items()}
                         for i in range(site + 1, c.num_hidden_layers)}

    def spec(self):
        return {"site": self.site, "rank": self.rank, "tau": self.tau, "slots": list(SLOTS)}

    def message(self, payload):
        """payload: {slot: (B, d) embedding-space vectors, unit RMS}. -> (B, d)."""
        out = 0.0
        for i, s in enumerate(SLOTS):
            out = out + self.slot_op.weight[i][None, :] + self.proj[s](payload[s])
        return out


def unit_rms(x):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def payload_from_ids(model, ids):
    """{slot: (B, K) token ids} -> {slot: (B, K, d)} unit-RMS frozen embeddings."""
    return {s: unit_rms(mx.stop_gradient(model.model.embed_tokens(v))) for s, v in ids.items()}


def prefix(model, tokens, site):
    """The original model's state after block `site` (identical on the bus path: the LoRA starts after the site)."""
    h = model.model.embed_tokens(tokens)
    for i in range(site + 1):
        h = model.model.layers[i](h)
    return h


def forward(model, tokens, port=None, mentions=None, payload=None, at=None, h_site=None):
    """mentions (B, K) positions (-1 = none); payload {slot: (B, K, d)}. port None = the original model. h_site: a
    precomputed prefix(model, tokens, port.site) to reuse."""
    if port is None:
        h = model.model.embed_tokens(tokens)
        for blk in model.model.layers:
            h = blk(h)
        return model.head(h if at is None else h[mx.arange(tokens.shape[0]), at])
    B, T = tokens.shape
    h = prefix(model, tokens, port.site) if h_site is None else h_site
    for i in range(port.site, len(model.model.layers)):
        if i > port.site:
            h = model.model.layers[i](h, "causal", port.lora.get(f"b{i}"))
        if i == port.site:
            for k in range(mentions.shape[1]):
                pos = mentions[:, k]
                ok = (pos >= 0).astype(h.dtype)[:, None]
                p0 = mx.maximum(pos, 0)
                h_row = h[mx.arange(B), p0].astype(mx.float32)
                msg = port.message({s: payload[s][:, k] for s in SLOTS})
                delta = (LB.bounded(port.gate * msg, h_row, port.tau) * ok).astype(h.dtype)
                mask = (mx.arange(T)[None, :] == p0[:, None]).astype(h.dtype)[..., None]
                h = h + mask * delta[:, None, :]
    return model.head(h if at is None else h[mx.arange(B), at])


def site_states(model, tokens, site, mentions, h=None):
    """The original model's state after block `site` at each mention: (B, K, d) (what a tile reads)."""
    h = prefix(model, tokens, site) if h is None else h
    B = tokens.shape[0]
    return mx.stack([h[mx.arange(B), mx.maximum(mentions[:, k], 0)] for k in range(mentions.shape[1])], axis=1)


# ------------------------------------------------------------------ prompts
def make_row(tok, voc, task, papers, rng, templates):
    """A realistic Q&A row: (prompt ids, answer ids, mentions, record ids, meta) or None (no valid pair / too long)."""
    if len(papers) == 2 and papers[0]["title"] == papers[1]["title"]:
        return None
    qa = QA.make_qa(task, papers, rng, templates, format_chat=getattr(tok, "chat", QA.chat))
    if qa is None:
        return None
    prompt, answer, titles, meta = qa
    p, a = QA.encode(tok, prompt, answer)
    if len(p) + len(a) > SEQ:
        return None
    ment = QA.mention_positions(tok, prompt, titles)
    rec = {s: [voc.ids[s][voc.values[s].index(pp[s])] for pp in papers[:len(titles)]] for s in SLOTS}
    return p, a, ment, rec, meta


def stack(tok, rows):
    """rows: [(p, a, ment, rec, meta)] -> tokens, loss mask, targets, mentions, records."""
    return QA.stack(tok, [r[:4] for r in rows])


def answer_loss(logits, W, Y):
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), Y, reduction="none")
    return mx.sum(ce * W) / mx.maximum(mx.sum(W), 1.0)


def generate(model, tok, port, prompts, mentions, payload, max_new=48):
    """Greedy batched generation, right-padded (causal attention never sees the pads). The record writes stay at the
    prompt's mention positions. port None = the original model."""
    B = len(prompts)
    L = max(len(p) for p in prompts) + max_new
    buf = [p + [tok.pad] * (L - len(p)) for p in prompts]
    cur = [len(p) for p in prompts]
    done = [False] * B
    ends = {tok.t.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>", "[EOS]")} - {None}   # [EOS]: Kimi
    for _ in range(max_new):
        Lb = min(L, -(-max(cur) // 16) * 16)           # bucketed length (pads past each row's end are never attended)
        T = mx.array(buf)[:, :Lb]
        at = mx.array([c - 1 for c in cur])
        logits = forward(model, T, port, mentions, payload, at=at) if port is not None else forward(model, T, None, at=at)
        nxt = mx.argmax(logits, -1).tolist()
        for b in range(B):
            if done[b]:
                continue
            buf[b][cur[b]] = nxt[b]
            cur[b] += 1
            done[b] = nxt[b] in ends
        if all(done):
            break
    return [tok.t.decode(buf[b][len(prompts[b]):cur[b]]) for b in range(B)]


def outside_titles(facts, tok, n=20000, seed=0):
    """Titles of harvest papers outside the knowledge world, for the mock provider: the same licence filter as the
    facts, every field, then a seeded shuffle stratified by field (review: the first version was 95% cond-mat)."""
    world = {p["id"] for p in facts["papers"]}
    by_field = {}
    for f in sorted(PF.HARVEST.glob("physics_*.jsonl")):
        base = f.stem.removeprefix("physics_")
        for line in f.open():
            r = json.loads(line)
            lic = r.get("license") or ""
            if r["id"] in world or not any(s in lic for s in PF.OK_LICENSES):
                continue
            t = re.sub(r"\s+", " ", r["title"]).strip()
            if "'" not in t and len(tok.encode(" '" + t + "'")) <= PF.TITLE_TOKENS_MAX:
                by_field.setdefault(base, []).append(t)
    rng = random.Random(30_000 + seed)
    per = n // len(by_field)
    out = []
    for k in sorted(by_field):
        ts = sorted(set(by_field[k]))
        rng.shuffle(ts)
        out += ts[:per]
    rng.shuffle(out)
    return out


def mock_paper(titles, voc, rng):
    return {"title": rng.choice(titles), **{s: rng.choice(voc.values[s]) for s in SLOTS}}


def balance_same_field(task, ps, rng, voc=None, pool=None, weights=None):
    """same_field pairs are Yes/No 50/50 (independent draws give ~89% No; review). Mock papers get their field set;
    world pairs are redrawn from the pool."""
    if task != "same_field":
        return ps
    want_same = rng.random() < 0.5
    A, B = ps
    if pool is None:
        B = dict(B, field=A["field"] if want_same else rng.choice([f for f in voc.values["field"] if f != A["field"]]))
        return [A, B]
    idx = [i for i, p in enumerate(pool) if (p["field"] == A["field"]) == want_same and p["id"] != A["id"]]
    # redraw B with the sampling weights (review: a uniform redraw gave rare papers 5x their share in this slot)
    w = [weights[i] for i in idx] if weights else None
    return [A, pool[rng.choices(idx, weights=w, k=1)[0]]]


def mock_rows(tok, voc, titles, rng, n, templates):
    rows = []
    while len(rows) < n:
        task = rng.choice(RETRIEVAL + REASONING)
        k = 1 if task in RETRIEVAL else 2
        ps = [mock_paper(titles, voc, rng) for _ in range(k)]
        if k == 2 and ps[0]["title"] == ps[1]["title"]:
            continue
        ps = balance_same_field(task, ps, rng, voc)
        r = make_row(tok, voc, task, ps, rng, templates)
        if r is not None:
            rows.append((task,) + r)
    return rows


# ------------------------------------------------------------------ M5 memory guards (coordinator, 2026-11-23)
GATE_GIB = 30.0            # free + inactive + purgeable before loading the model
KIMI_GATE_GIB = 75.0        # Kimi Linear 8-bit: 49 GB of weights + training headroom (prior mlx2 Kimi LoRA peaks 50-70 GiB)
SWAP_GROWTH_GIB = 1.0      # after warm-up: stop cleanly if swap-outs grow by more than this from the baseline
SWAP_WARMUP_STEPS = 20     # coordinator 2026-11-23: baseline after the first 20 training steps of each launch
SWAP_WARMUP_GIB = 3.0      # ... but stop if the warm-up itself adds more than this


def _vm():
    import subprocess
    out = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    vals = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(.+?):\s+(\d+)\.", out, re.M)}
    return page, vals


def available_gib():
    page, v = _vm()
    return (v.get("Pages free", 0) + v.get("Pages inactive", 0) + v.get("Pages purgeable", 0)) * page / 2 ** 30


def swapouts_gib():
    page, v = _vm()
    return v.get("Swapouts", 0) * page / 2 ** 30


def gate_gib(args) -> float:
    """Memory needed before loading: GATE_GIB for Qwen3-4B; the Kimi Linear 8-bit weights alone are 49 GB."""
    return KIMI_GATE_GIB if getattr(args, "backend", "qwen3") == "kimi" else GATE_GIB


def memory_gate(wait_s: int = 600, step_s: int = 30, gib: float | None = None):
    """Before loading: wait until >= GATE_GIB is available (back off and retry); exit 3 (a clean, resumable stop) if
    it never is within wait_s, so the slot is released rather than pushing the host into swap."""
    t0 = time.time()
    while True:
        a = available_gib()
        if a >= (GATE_GIB if gib is None else gib):
            print(f"memory gate ok: {a:.1f} GiB available", flush=True)
            return
        if time.time() - t0 > wait_s:
            print(f"memory gate: only {a:.1f} GiB available after {wait_s} s; stopping", flush=True)
            sys.exit(3)
        time.sleep(step_s)


class SwapGuard:
    """Coordinator rule (2026-11-23): the first SWAP_WARMUP_STEPS training steps of a launch may page out up to
    SWAP_WARMUP_GIB (one-time allocation); the baseline is then reset and growth above SWAP_GROWTH_GIB stops the chunk.
    warmup_gib is logged in each launch's receipt log."""

    def __init__(self):
        self.load = swapouts_gib()
        self.base = None
        self.warmup_gib = None

    def check(self, steps_this_launch: int):
        """-> None (ok) or a reason string (stop cleanly)."""
        now = swapouts_gib()
        if self.base is None:
            grown = now - self.load
            if grown > SWAP_WARMUP_GIB:
                self.warmup_gib = round(grown, 3)
                return f"swap warm-up growth {grown:.2f} GiB over {SWAP_WARMUP_GIB} GiB"
            if steps_this_launch >= SWAP_WARMUP_STEPS:
                self.base, self.warmup_gib = now, round(grown, 3)
            return None
        if now - self.base > SWAP_GROWTH_GIB:
            return f"swap growth {now - self.base:.2f} GiB over {SWAP_GROWTH_GIB} GiB after warm-up"
        return None


# ------------------------------------------------------------------ resume (autosave + TERM, as loop_train)
_TERM = {"requested": False}
_FREE = {"max_seconds": None, "no_memory_gate": None, "chunk_tag": None}
_ARG_DEFAULTS = {"backend": "qwen3", "kda_policy": "chunk_wy"}


def _on_term(signum, frame):
    _TERM["requested"] = True


def save_state(out: Path, tree: dict, opt, it, rngs, log, t_used):
    tmp, cur, old = out / "resume.tmp", out / "resume", out / "resume.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    mx.save_safetensors(str(tmp / "model.safetensors"), dict(tree_flatten(tree)))
    mx.save_safetensors(str(tmp / "opt.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "stream.pkl").write_bytes(pickle.dumps({"it": it, "rngs": [r.getstate() for r in rngs], "log": log,
                                                   "train_seconds": t_used}))
    for f in list(tmp.iterdir()) + [tmp]:
        fd = os.open(f, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    if cur.exists():
        if old.exists():
            shutil.rmtree(old)
        cur.rename(old)
    tmp.rename(cur)
    if old.exists():
        shutil.rmtree(old)


def resume_dir(out: Path):
    for n in ("resume", "resume.old"):
        if (out / n / "stream.pkl").exists():
            return out / n
    return None


def load_state(d: Path, module, opt, rngs):
    s = pickle.loads((d / "stream.pkl").read_bytes())
    module.load_weights(str(d / "model.safetensors"))
    opt.state = tree_unflatten(list(mx.load(str(d / "opt.safetensors")).items()))
    for r, st in zip(rngs, s["rngs"]):
        r.setstate(st)
    return s["it"], s["log"], s["train_seconds"]


# ------------------------------------------------------------------ adapter training
CACHE_LIMIT_GIB = 2.0      # MLX buffer cache cap (2026-11-23: variable shapes grew the cache and pushed the host into swap)


def mem_stats():
    return {"active_gib": round(mx.get_active_memory() / 2 ** 30, 2), "cache_gib": round(mx.get_cache_memory() / 2 ** 30, 2),
            "peak_gib": round(mx.get_peak_memory() / 2 ** 30, 2)}


def load_world(args):
    mx.set_cache_limit(int(CACHE_LIMIT_GIB * 2 ** 30))
    facts = json.loads(Path(args.facts).read_text())
    if getattr(args, "backend", "qwen3") == "kimi":       # runs before --backend existed are Qwen3
        from research.ens import kimi_core as KC
        model = KC.KimiCore(args.model, kda_policy=getattr(args, "kda_policy", "chunk_wy"))
        tok = KC.KimiTok(model)
        return facts, model, model.config, model.checkpoint, tok, Vocab(tok, facts, model)
    path = Q.snapshot_dir(args.model)
    model, cfg, ckpt = Q.load(path, dtype=mx.bfloat16, repo=args.model)
    model.freeze()
    tok = LB.Tok(path)
    return facts, model, cfg, ckpt, tok, Vocab(tok, facts)


def train(args) -> int:
    out = Path(args.out)
    if (out / "receipt.json").exists():
        print("already finished", flush=True)
        return 0
    resuming = resume_dir(out) is not None
    if not resuming:
        if out.is_dir() and [p.name for p in out.iterdir()] == ["args.json"]:
            (out / "args.json").unlink()
        out = C.fresh_out(args.out)
        C.write_json(out / "args.json", vars(args))
    elif _ARG_DEFAULTS | json.loads((out / "args.json").read_text()) | _FREE != vars(args) | _FREE:
        # runs started before --backend / --kda-policy existed are Qwen3 runs with the defaults (review)
        raise SystemExit("resume: arguments differ")
    mx.random.seed(args.seed)
    if not args.no_memory_gate:
        memory_gate(gib=gate_gib(args))
    facts, model, cfg, ckpt, tok, voc = load_world(args)
    guard = SwapGuard()
    port = RecordPort(model, args.site, args.rank, args.tau)
    titles = outside_titles(facts, tok, seed=args.seed)
    text = LB.text_windows(tok, "train")
    rng, jrng = random.Random(10_000 + args.seed), random.Random(20_000 + args.seed)
    wu = max(1, min(100, args.steps // 10))
    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, wu), optim.cosine_decay(args.lr, max(1, args.steps - wu), args.lr * 0.1)],
        [wu]), weight_decay=0.0)
    opt.init(port.trainable_parameters())
    it0, log, t_used = 0, [], 0.0
    if resuming:
        it0, log, t_used = load_state(resume_dir(out), port, opt, [rng, jrng])
        print(f"resumed at {it0}", flush=True)

    def lf(p, T, W, Y, M, R, ttok, tfield, tR, tref):
        ce = answer_loss(forward(model, T, p, M, payload_from_ids(model, R)), W, Y)
        tl = forward(model, ttok, p, tfield, payload_from_ids(model, tR)).astype(mx.float32)
        kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1)), -1))
        return ce + args.kl_weight * kl, (ce, kl)
    g = nn.value_and_grad(port, lambda p, *a: lf(p, *a)[0])
    n_text = max(1, int(round(args.batch * args.p_text)))
    _TERM["requested"] = False
    old = signal.signal(signal.SIGTERM, _on_term)
    t0, it = time.time(), it0
    try:
        while it < args.steps:
            n_launch = it - it0
            swap = None
            if not args.no_memory_gate and (n_launch <= SWAP_WARMUP_STEPS or n_launch % 10 == 0):
                had_base = guard.base is not None
                swap = guard.check(n_launch)
                if guard.base is not None and not had_base:
                    log.append({"step": it, "swap_warmup_gib": guard.warmup_gib})
                    print(log[-1], flush=True)
            if _TERM["requested"] or swap or (args.max_seconds and time.time() - t0 > args.max_seconds):
                if swap:
                    log.append({"step": it, "swap_stop": swap, "swap_warmup_gib": guard.warmup_gib})
                save_state(out, port.parameters(), opt, it, [rng, jrng], log, t_used + time.time() - t0)
                print(f"stopped at {it} (resume state saved){' - ' + swap if swap else ''}", flush=True)
                return 3
            it += 1
            rows = mock_rows(tok, voc, titles, rng, args.batch - n_text, T_TRAIN)
            T, W, Y, M, R = stack(tok, [r[1:] for r in rows])
            tw = [text[rng.randrange(len(text))] for _ in range(n_text)]
            ttok = mx.array(tw)
            tfield = mx.array([[rng.randrange(8, 16), -1] for _ in range(n_text)])
            tR = {s: mx.array([[voc.ids[s][rng.randrange(len(voc.ids[s]))], 0] for _ in range(n_text)]) for s in SLOTS}
            tref = mx.stop_gradient(model(ttok).astype(mx.float32))
            loss, grads = g(port, T, W, Y, M, R, ttok, tfield, tR, tref)
            opt.update(port, grads)
            mx.eval(port.parameters(), opt.state)
            if it % args.log_every == 0 or it == args.steps:
                log.append({"step": it, "loss": round(float(loss.item()), 5),
                            "elapsed_s": round(t_used + time.time() - t0, 1), **mem_stats()})
                print(log[-1], flush=True)
            if args.save_every and it % args.save_every == 0 and it < args.steps:
                save_state(out, port.parameters(), opt, it, [rng, jrng], log, t_used + time.time() - t0)
        if not (out / "port.safetensors").exists():
            save_state(out, port.parameters(), opt, it, [rng, jrng], log, t_used + time.time() - t0)
    finally:
        signal.signal(signal.SIGTERM, old)
    port.freeze()
    wpath = out / "port.safetensors"
    if not wpath.exists():                             # written once, before evaluation (review)
        mx.save_safetensors(str(wpath), dict(tree_flatten(port.parameters())))
    with wpath.open("rb") as fh:
        pfp = hashlib.file_digest(fh, "sha256").hexdigest()[:12]
    receipt = {"schema": "ens.h43.physics_bus_receipt.v1", "recipe": RECIPE, "base_checkpoint": ckpt,
               "port_checkpoint": f"ens-h43-recordport-{args.model.split('/')[-1].lower()}-site{args.site}-s{args.seed}@{pfp}",
               "port": port.spec(), "facts_sha256": facts["sha256"], "facts_label": facts["label"], "args": vars(args),
               "trainable_params": C.n_params(port), "templates": {"train": T_TRAIN, "held": T_HELD},
               "log": log, **C.source_revision(), **C.host_record()}
    C.write_json(out / "train_receipt.json", receipt)      # tiles can start from this; evaluation resumes by cell
    deadline = (t0 + args.max_seconds) if args.max_seconds else None
    receipt["eval"] = evaluate(model, port, tok, voc, facts, titles, args, EvalCells(out, deadline))
    C.write_json(out / "receipt.json", receipt)
    print("eval", json.dumps(receipt["eval"]), flush=True)
    return 0


def eval_batches(rows):
    """Keep every row pairable for the swapped-record control, even at 16k + 1 rows."""
    start = 0
    while start < len(rows):
        size = 15 if len(rows) - start == 17 else 16
        yield rows[start:start + size]
        start += size


def score_batch(tasks, texts, metas, concepts=()):
    return [QA.score(tk, tx, m, concepts) for tk, tx, m in zip(tasks, texts, metas)]


class EvalCells:
    """Resumable evaluation (review): each finished cell is written atomically to <out>/eval_cells.json; a deadline is
    checked between batches; a rerun skips finished cells."""

    def __init__(self, out: Path, deadline: float | None):
        self.path, self.deadline = out / "eval_cells.json", deadline
        self.cells = json.loads(self.path.read_text()) if self.path.exists() else {}

    def done(self, key):
        return key in self.cells

    def put(self, key, value):
        self.cells[key] = value
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cells))
        tmp.replace(self.path)

    def check(self):
        if self.deadline and time.time() > self.deadline:
            print("evaluation paused at the deadline (finished cells saved)", flush=True)
            raise SystemExit(3)


def evaluate(model, port, tok, voc, facts, titles, args, cells: EvalCells):
    """Adapter ceiling on mock records over outside titles (train / held questions): answers generated freely and
    parsed (physics_qa.score); the base model on the same questions (no record); records rotated to another row
    (if answers stay correct they did not come from the record); held-text drift. Example generations are kept."""
    concepts = voc.values["concept"]
    for name, temps in (("train_questions", T_TRAIN), ("held_questions", T_HELD)):
        for task in RETRIEVAL + REASONING:
            key = f"{name}.{task}"
            if cells.done(key):
                continue
            rng = random.Random(70_000 + args.seed + hash_task(name, task))
            rows = []
            while len(rows) < args.eval_n:
                k = 1 if task in RETRIEVAL else 2
                ps = balance_same_field(task, [mock_paper(titles, voc, rng) for _ in range(k)], rng, voc)
                r = make_row(tok, voc, task, ps, rng, temps)
                if r is not None:
                    rows.append(r)
            tally, ex = {}, None
            for sub in eval_batches(rows):
                cells.check()
                prompts = [r[0] for r in sub]
                M = mx.array([r[2] + [-1] * (2 - len(r[2])) for r in sub])
                R = {s: mx.array([r[3][s] + [0] * (2 - len(r[3][s])) for r in sub]) for s in SLOTS}
                Rs = {s: mx.concatenate([R[s][1:], R[s][:1]]) for s in SLOTS} if len(sub) > 1 else None
                metas = [r[4] for r in sub]
                mn = QA.MAX_NEW[task]
                outs = {"base": generate(model, tok, None, prompts, M, None, mn),
                        "record": generate(model, tok, port, prompts, M, payload_from_ids(model, R), mn)}
                if Rs is not None:
                    outs["swapped_record"] = generate(model, tok, port, prompts, M, payload_from_ids(model, Rs), mn)
                for arm, texts in outs.items():
                    for sc in score_batch([task] * len(sub), texts, metas, concepts):
                        for kk, v in sc.items():
                            tally[f"{arm}.{kk}"] = tally.get(f"{arm}.{kk}", 0) + int(v)
                if ex is None:
                    ex = [{"q": tok.t.decode(prompts[i]), "record": outs["record"][i], "base": outs["base"][i]}
                          for i in range(min(2, len(sub)))]
            cells.put(key, {k: round(v / len(rows), 4) for k, v in sorted(tally.items())} | {"n": len(rows),
                                                                                           "examples": ex})
    if not cells.done("held_text"):
        held = LB.text_windows(tok, "held")
        rng = random.Random(80_000 + args.seed)
        tw = [held[rng.randrange(len(held))] for _ in range(128)]
        s = [0.0, 0.0, 0]
        for a0 in range(0, len(tw), 16):
            tt = mx.array(tw[a0:a0 + 16])
            B = tt.shape[0]
            fld = mx.array([[rng.randrange(8, 16), -1] for _ in range(B)])
            R = {sl: mx.array([[voc.ids[sl][rng.randrange(len(voc.ids[sl]))], 0] for _ in range(B)]) for sl in SLOTS}
            ref = model(tt).astype(mx.float32)
            adm = forward(model, tt, port, fld, payload_from_ids(model, R)).astype(mx.float32)
            for i, lg in enumerate((ref, adm)):
                s[i] += float(nn.losses.cross_entropy(lg[:, :-1], tt[:, 1:], reduction="sum").item())
            s[2] += B * (tt.shape[1] - 1)
        cells.put("held_text", {"nll_original": round(s[0] / s[2], 5), "nll_admitted": round(s[1] / s[2], 5),
                                "nll_cost": round((s[1] - s[0]) / s[2], 5)})
    return dict(cells.cells)


def hash_task(*parts):
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:6], 16)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--facts", required=True)
    p.add_argument("--model", default=Q.QWEN3_4B, help="HF repo (qwen3) or local path (kimi)")
    p.add_argument("--backend", choices=("qwen3", "kimi"), default="qwen3",
                   help="kimi: mlx2's Kimi Linear 48B-A3B through research/ens/kimi_core.py")
    p.add_argument("--kda-policy", choices=("reference", "chunk_wy"), default="chunk_wy",
                   help="kimi: KDA arithmetic (both MLX ops with a checked backward; chunk_wy is faster)")
    p.add_argument("--site", type=int, default=10)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--p-text", type=float, default=0.25)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=128)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--no-memory-gate", action="store_true", help="skip the M5 memory gate and swap guard (CPU tests)")
    p.add_argument("--chunk-tag", default=None, help="ignored; lets the M5 slot watchdog find this process")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.backend == "kimi" and args.model == Q.QWEN3_4B:
        from research.ens import kimi_core as KC
        args.model = KC.KIMI_LINEAR_8BIT                 # review: the Qwen default must not reach the Kimi loader
    sys.exit(train(args))


if __name__ == "__main__":
    main()
