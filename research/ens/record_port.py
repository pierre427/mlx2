"""ENS H54 record port v2: records from a system of record read by a frozen Qwen3 core through a slot bank.

Pierre (2026-11-24, via the coordinator): "Can we make a tile that acts as an interface between a system of record
(markdown, psql db, etc) like system-of-record <-> record port tile <-> bus <-> kimi"; "try it yes". Design:
wiki/docs/ens/h54-record-port.md. This module is "port v2" plus the eval harness for the arms base / pasted text /
port v2 / swapped records (the KV and HS arms are separate).

Port v2 (exists only on the port path; with no records, or at initialisation, the model is the original bit for bit):

  slot bank   the records admitted to a prompt (sor.Record list). Each value is tokenised as it would appear in an
              answer (leading space); one slot per value token (at most MAX_POS). Slot key =
              Wa(attr) + We(entity) + Wp(prev) + pos[j], with attr / entity = the mean frozen embedding of the
              attribute / entity name tokens (unit RMS), prev = the frozen embedding of the previous value token (a
              learned start vector at j = 0), pos a learned table; then RMS-normed. Slot value = the value token's
              frozen embedding (unit RMS). Nothing indexes the record's place in the list, so record order carries no
              information (attention over slots is permutation invariant).
  reader      after each of the next L blocks after `site`: h += s * g_i * O_i(softmax(Q_i(norm(h)) K_i(key)^T / sqrt(r))
              V_i(value)), rank r, with g_i initialised to 0 (s = a fixed gate scale, so Adam moves the gate at s * lr
              per step instead of lr; recorded in the port spec). Rows with no records get exactly 0.
  LoRA        rank-16 LoRA on the blocks after the site, as physics_bus; masked per row, so a row without records is
              the original model even inside a mixed batch.

Deviations from the design text:
  - by the coordinator's brief, the entity part of the key is the mean frozen embedding of the entity's name (not its
    site state at the mention), so the bank needs no mention positions and is built before the prompt runs;
  - records per prompt are 1, 4 and 16 (NRECS; the design also lists 64), and there are no entities-per-question
    cells (1 / 2 / 4): lookups ask one entity and compares two, as sor_mock draws them.

Data: sor_mock.sample(seed, index, split="train") (fresh random schema per row; deterministic by (seed, index), so a
resumed run replays exactly the same stream), Qwen3 chat prompt with the question, the full-sentence answer as the
target, loss on answer tokens only. The loss is logged in two parts: value tokens (the asked value, or the winning
entity of a compare, inside the answer) and frame tokens (the rest of the sentence). A falling total with a value loss
at chance means the port learned the frames and copies nothing (review, 2026-11-24); --value-loss-weight upweights the
value tokens. Optional ordinary-text rows (--p-text, physics_bus' corpus windows) with records admitted carry a KL to
the original model.

Arms: base (no records), pasted (sor_mock.pasted_context, no port), port, swapped (the same entities and attributes
with every value redrawn to a different value of the same type: swapped.strict = the old value is still emitted,
swapped.follows_record = the new value is emitted and the old one is absent), irrelevant (another sample's records).

Kill measures (design, fixed before any run), computed into eval_summary.json on held schemas x held frames:
  - port v2 below 0.9 strict on IDs, numbers and dates (lookup + injection);
  - port more than 0.05 below pasted text at the same records;
  - swapped records above 0.2 strict on the old value (lookup + injection; compare is reported against its 0.5
    chance separately);
  - ordinary-text NLL cost above 0.05 nats (records admitted).
  A measure that could not be computed makes the verdict "inconclusive", never "not killed". "no case": pasted text
  matches or beats the port at every record count and the port's measured prefill is not faster than pasted text.

  python -m research.ens.record_port train --model Qwen/Qwen3-4B --site 10 --steps 3000 --batch 16 --out DIR
  python -m research.ens.record_port eval --run DIR [--n 24]

Resumable for 20-minute M5 slots (autosave, SIGTERM -> save + exit 3, --max-seconds), with physics_bus' memory gate
and swap guard; evaluation resumes by cell (eval_cells.json) and pauses at the deadline with exit 3.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import lm_bus as LB
from research.ens import physics_bus as PB
from research.ens import physics_qa as QA
from research.ens import qwen3 as Q
from research.ens import sor_mock as M

RECIPE = "ens.h54.record_port.v2"
MAX_POS = 32                     # slots per value (longer values are truncated and counted)
TASKS = ("lookup", "compare", "injection")
NRECS = (1, 4, 16)
KILL = {"held_exact": 0.9, "vs_pasted": 0.05, "swapped": 0.2, "text_nll": 0.05}
_FREE = {"max_seconds": None, "no_memory_gate": None, "chunk_tag": None, "stop_after": None}


def end_text(tok) -> str:
    return getattr(tok, "end_text", "<|im_end|>")


# ------------------------------------------------------------------ slot bank
@dataclass
class Bank:
    """Padded across the batch. R = records, S = slots, La / Le = name tokens. Masks are float32 0/1."""
    attr: mx.array        # (B, R, La) attribute-name token ids
    attr_m: mx.array
    ent: mx.array         # (B, R, Le) entity-name token ids
    ent_m: mx.array
    slot_rec: mx.array    # (B, S) record of each slot (an index into this row's record list, never a feature)
    slot_val: mx.array    # (B, S) value token id
    slot_prev: mx.array   # (B, S) previous value token id (0 at j = 0, replaced by the start vector)
    slot_pos: mx.array    # (B, S) position j within the value
    slot_m: mx.array      # (B, S)
    has: mx.array         # (B,) 1 if the row has any slot
    n_slots: list         # per row
    truncated: int        # values cut at MAX_POS
    any: bool


def _bucket(n, b):
    return max(b, -(-n // b) * b)


def build_bank(tok, record_lists, max_pos: int = MAX_POS) -> Bank:
    """record_lists: per row, a list of sor.Record (may be empty)."""
    cache = {}

    def enc(s):
        if s not in cache:
            cache[s] = tok.encode(" " + s)
        return cache[s]
    rows, trunc = [], 0
    for recs in record_lists:
        rr, slots = [], []
        for k, r in enumerate(recs):
            v = enc(r.value)
            if len(v) > max_pos:
                trunc += 1
                v = v[:max_pos]
            rr.append((enc(r.attribute) or [0], enc(r.entity) or [0]))
            for j, t in enumerate(v):
                slots.append((k, t, v[j - 1] if j else 0, j))
        rows.append((rr, slots))
    R = _bucket(max((len(r[0]) for r in rows), default=0), 8)
    S = _bucket(max((len(r[1]) for r in rows), default=0), 16)
    La = max([len(a) for rr, _ in rows for a, _ in rr] or [1])
    Le = max([len(e) for rr, _ in rows for _, e in rr] or [1])
    attr, attr_m, ent, ent_m = [], [], [], []
    srec, sval, sprev, spos, sm = [], [], [], [], []
    for rr, slots in rows:
        pad_r = [([0], [0])] * (R - len(rr))
        a_rows = [a for a, _ in rr] + [[] for _ in pad_r]
        e_rows = [e for _, e in rr] + [[] for _ in pad_r]
        attr.append([a + [0] * (La - len(a)) for a in a_rows])
        attr_m.append([[1] * len(a) + [0] * (La - len(a)) for a in a_rows])
        ent.append([e + [0] * (Le - len(e)) for e in e_rows])
        ent_m.append([[1] * len(e) + [0] * (Le - len(e)) for e in e_rows])
        pad = [(0, 0, 0, 0)] * (S - len(slots))
        srec.append([s[0] for s in slots + pad])
        sval.append([s[1] for s in slots + pad])
        sprev.append([s[2] for s in slots + pad])
        spos.append([s[3] for s in slots + pad])
        sm.append([1] * len(slots) + [0] * len(pad))
    n_slots = [len(r[1]) for r in rows]
    f32 = mx.float32
    return Bank(mx.array(attr), mx.array(attr_m).astype(f32), mx.array(ent), mx.array(ent_m).astype(f32),
                mx.array(srec), mx.array(sval), mx.array(sprev), mx.array(spos), mx.array(sm).astype(f32),
                mx.array([1.0 if n else 0.0 for n in n_slots]), n_slots, trunc, any(n_slots))


# ------------------------------------------------------------------ port v2
class Reader(nn.Module):
    def __init__(self, d, dk, r, gate_scale):
        super().__init__()
        self.r, self.gate_scale = r, gate_scale
        self.norm = nn.RMSNorm(d)
        self.q = nn.Linear(d, r, bias=False)
        self.k = nn.Linear(dk, r, bias=False)
        self.v = nn.Linear(d, r, bias=False)
        self.o = nn.Linear(r, d, bias=False)
        self.gate = mx.zeros((1,))                       # closed at init: the output is exactly 0

    def __call__(self, h, keys, vals, bias, m):
        q = self.q(self.norm(h.astype(mx.float32)))                       # (B, T, r)
        s = (q @ self.k(keys).transpose(0, 2, 1)) * (self.r ** -0.5) + bias[:, None, :]
        out = self.o(mx.softmax(s, axis=-1) @ self.v(vals))              # (B, T, d)
        return h + (self.gate * self.gate_scale * out).astype(h.dtype) * m


class RecordPortV2(nn.Module):
    def __init__(self, model, site: int, readers: int = -1, rank: int = 16, reader_rank: int = 64,
                 key_dim: int = 512, gate_scale: float = 10.0, max_pos: int = MAX_POS):
        super().__init__()
        c = model.config
        d, n = c.hidden_size, c.num_hidden_layers
        if not 0 <= site < n - 1:
            raise ValueError(f"site {site} must leave at least one block after it (n = {n})")
        L = n - 1 - site if readers < 0 else min(readers, n - 1 - site)
        self.site, self.n_readers, self.rank, self.reader_rank = site, L, rank, reader_rank
        self.key_dim, self.gate_scale, self.max_pos = key_dim, gate_scale, max_pos
        self.wa = nn.Linear(d, key_dim, bias=False)
        self.we = nn.Linear(d, key_dim, bias=False)
        self.wp = nn.Linear(d, key_dim, bias=False)
        self.pos = nn.Embedding(max_pos, key_dim)
        self.start = mx.random.normal((key_dim,)) * 0.02
        self.key_norm = nn.RMSNorm(key_dim)
        self.readers = {f"b{i}": Reader(d, key_dim, reader_rank, gate_scale) for i in range(site + 1, site + 1 + L)}
        if hasattr(model, "lora_dims"):                  # Kimi-style core: per-layer projection shapes
            dims_of = model.lora_dims
        else:
            H, K, hd, I = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.intermediate_size
            dims = {"q_proj": (d, H * hd), "k_proj": (d, K * hd), "v_proj": (d, K * hd), "o_proj": (H * hd, d),
                    "gate_proj": (d, I), "up_proj": (d, I), "down_proj": (I, d)}
            dims_of = lambda i: dims
        self.lora = {f"b{i}": {k: LB.LoRA(a, b, rank) for k, (a, b) in dims_of(i).items()}
                     for i in range(site + 1, n)}

    def spec(self):
        return {"site": self.site, "readers": self.n_readers, "rank": self.rank, "reader_rank": self.reader_rank,
                "key_dim": self.key_dim, "gate_scale": self.gate_scale, "max_pos": self.max_pos,
                "key": "wa(attr)+we(entity)+wp(prev|start)+pos[j]", "entity": "mean frozen name embedding"}

    def encode_bank(self, model, bank: Bank):
        """-> keys (B, S, dk), values (B, S, d), additive mask bias (B, S); float32."""
        E = model.model.embed_tokens
        emb = lambda ids: mx.stop_gradient(E(ids)).astype(mx.float32)

        def mean_name(ids, m):
            s = mx.sum(emb(ids) * m[..., None], axis=2) / mx.maximum(mx.sum(m, axis=2, keepdims=True), 1.0)
            return PB.unit_rms(s)                                           # (B, R, d)
        attr, ent = mean_name(bank.attr, bank.attr_m), mean_name(bank.ent, bank.ent_m)
        B, R, d = attr.shape
        idx = (bank.slot_rec + mx.arange(B)[:, None] * R).reshape(-1)
        a_s = attr.reshape(B * R, d)[idx].reshape(B, -1, d)
        e_s = ent.reshape(B * R, d)[idx].reshape(B, -1, d)
        prev = mx.where((bank.slot_pos == 0)[..., None], self.start, self.wp(PB.unit_rms(emb(bank.slot_prev))))
        keys = self.key_norm(self.wa(a_s) + self.we(e_s) + prev + self.pos(bank.slot_pos))
        vals = PB.unit_rms(emb(bank.slot_val))
        return keys, vals, (1.0 - bank.slot_m) * -1e9

    def lora_for(self, i, m):
        d = self.lora.get(f"b{i}")
        if d is None:
            return None
        return {k: (lambda x, f=f: f(x) * m) for k, f in d.items()}


def forward(model, tokens, port=None, bank: Bank | None = None, at=None):
    """port None, or no slot in the whole batch: the original model (physics_bus.forward without a port)."""
    if port is None or bank is None or not bank.any:
        return PB.forward(model, tokens, None, at=at)
    B = tokens.shape[0]
    keys, vals, bias = port.encode_bank(model, bank)
    h = PB.prefix(model, tokens, port.site)
    m = bank.has.astype(h.dtype)[:, None, None]
    for i in range(port.site + 1, len(model.model.layers)):
        h = model.model.layers[i](h, "causal", port.lora_for(i, m))
        r = port.readers.get(f"b{i}")
        if r is not None:
            h = r(h, keys, vals, bias, m)
    return model.head(h if at is None else h[mx.arange(B), at])


# ------------------------------------------------------------------ rows
def prompt_text(s: M.Sample, arm: str) -> str:
    return QA.chat(M.pasted_context(s) if arm == "pasted" else s.prompt)


def encode_row(tok, s: M.Sample, arm: str = "port"):
    return tok.encode(prompt_text(s, arm)), tok.encode(s.answer + end_text(tok))


def stack_rows(tok, items):
    """items: [(prompt ids, answer ids)] -> tokens (B, L), loss mask (B, L) on answer targets, targets (B, L)."""
    L = _bucket(max(len(p) + len(a) for p, a in items), 16)
    T, W, Y = [], [], []
    for p, a in items:
        seq = p + a
        T.append(seq + [tok.pad] * (L - len(seq)))
        Y.append((seq[1:] + [tok.pad] * (L - len(seq) + 1))[:L])
        W.append(([0] * (len(p) - 1) + [1] * len(a) + [0] * (L - len(seq) + 1))[:L])
    return mx.array(T), mx.array(W).astype(mx.float32), mx.array(Y)


def value_span(tok, s: M.Sample, a_ids):
    """(start, length) of the copied tokens inside the answer ids: the asked value (lookup / injection) or the winning
    entity (compare). None if they cannot be located exactly."""
    i = s.answer.find(s.expected)
    if i < 0:
        return None
    for pre_text, val_text in (((s.answer[:i - 1], " " + s.expected),) if i and s.answer[i - 1] == " " else ()) + (
            (s.answer[:i], s.expected),):
        pre, val = tok.encode(pre_text), tok.encode(val_text)
        if a_ids[len(pre):len(pre) + len(val)] == val:
            return len(pre), len(val)
    enc = getattr(getattr(tok, "t", None), "encode", None)
    try:
        offs = enc(s.answer + end_text(tok), add_special_tokens=False).offsets
    except Exception:                    # noqa: BLE001 - a tokenizer without offsets: report a miss
        return None
    hit = [k for k, (a, b) in enumerate(offs) if a < i + len(s.expected) and b > i]
    return (hit[0], len(hit)) if hit else None


def stack_train(tok, samples, arm="port"):
    """-> tokens, answer mask, value mask (the copied tokens, a subset of the answer mask), targets, misses."""
    items = [encode_row(tok, s, arm) for s in samples]
    T, W, Y = stack_rows(tok, items)
    L, Wv, miss = T.shape[1], [], 0
    for s, (p, a) in zip(samples, items):
        sp = value_span(tok, s, a)
        miss += sp is None
        row = [0] * L
        if sp is not None:
            for k in range(sp[0], sp[0] + sp[1]):
                row[len(p) - 1 + k] = 1          # position len(p)-1+k predicts answer token k
        Wv.append(row)
    return T, W, mx.array(Wv).astype(mx.float32), Y, miss


def answer_losses(logits, W, Wv, Y, value_weight=1.0):
    """(objective, frame-token CE, value-token CE). value_weight 1 = the plain answer-token mean."""
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), Y, reduction="none")
    Wf = W - Wv
    sf, sv, nf, nv = mx.sum(ce * Wf), mx.sum(ce * Wv), mx.sum(Wf), mx.sum(Wv)
    obj = (sf + value_weight * sv) / mx.maximum(nf + value_weight * nv, 1.0)
    return obj, sf / mx.maximum(nf, 1.0), sv / mx.maximum(nv, 1.0)


class Count:
    """A resumable counter, saved with the rng states (physics_bus.save_state calls getstate / setstate)."""

    def __init__(self):
        self.n = 0

    def getstate(self):
        return self.n

    def setstate(self, st):
        self.n = st


def train_samples(seed, it, n, pool=0):
    """The deterministic training stream: step `it` (1-based) takes indices (it-1)*n .. it*n-1 (mod pool)."""
    out = []
    for b in range(n):
        i = (it - 1) * n + b
        out.append(M.sample(seed, i % pool if pool else i, split="train"))
    return out


def text_windows(tok, split):
    """physics_bus / lm_bus ordinary English windows (32 tokens); a seam for tests."""
    return LB.text_windows(tok, split)


def text_records(seed, k, n):
    """Records to admit on ordinary-text rows: from a separate index range of the mock (train schemas)."""
    return [M.sample(seed, 10 ** 9 + k * n + b, split="train").records for b in range(n)]


# ------------------------------------------------------------------ model
def load_core(model_arg: str):
    """(model, checkpoint '<name>@<fingerprint>', tokenizer). model_arg: an HF repo in the cache or a local dir."""
    mx.set_cache_limit(int(PB.CACHE_LIMIT_GIB * 2 ** 30))
    p = Path(model_arg)
    path = p if (p / "config.json").exists() else Q.snapshot_dir(model_arg)
    model, _, ckpt = Q.load(path, dtype=mx.bfloat16, repo=model_arg)
    model.freeze()
    return model, ckpt, LB.Tok(path)


def make_port(model, a) -> RecordPortV2:
    return RecordPortV2(model, a["site"], a["readers"], a["rank"], a["reader_rank"], a["key_dim"], a["gate_scale"],
                        a["max_pos"])


# ------------------------------------------------------------------ training
def train(args) -> int:
    out = Path(args.out)
    if (out / "train_receipt.json").exists():
        print("already finished", flush=True)
        return 0
    resuming = PB.resume_dir(out) is not None
    if not resuming:
        if out.is_dir() and [p.name for p in out.iterdir()] == ["args.json"]:
            (out / "args.json").unlink()
        out = C.fresh_out(args.out)
        C.write_json(out / "args.json", vars(args))
    elif json.loads((out / "args.json").read_text()) | _FREE != vars(args) | _FREE:
        raise SystemExit("resume: arguments differ")
    if not args.no_memory_gate:
        PB.memory_gate()
    model, ckpt, tok = load_core(args.model)
    guard = None if args.no_memory_gate else PB.SwapGuard()
    mx.random.seed(args.seed)
    port = make_port(model, vars(args))
    n_text = round(args.batch * args.p_text)
    nq = args.batch - n_text
    text = text_windows(tok, "train") if n_text else None
    trng = random.Random(20_000 + args.seed)
    trunc, vmiss = Count(), Count()
    state = [trng, trunc, vmiss]
    wu = max(1, min(100, args.steps // 10))
    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, wu), optim.cosine_decay(args.lr, max(1, args.steps - wu), args.lr * 0.1)],
        [wu]), weight_decay=0.0)
    opt.init(port.trainable_parameters())
    it0, log, t_used = 0, [], 0.0
    if resuming:
        it0, log, t_used = PB.load_state(PB.resume_dir(out), port, opt, state)
        print(f"resumed at {it0}", flush=True)
    held = {}

    def lf(p):
        T, W, Wv, Y, bank = held["qa"]
        obj, lf_, lv = answer_losses(forward(model, T, p, bank), W, Wv, Y, args.value_loss_weight)
        if "text" not in held:
            return obj, lf_, lv
        ttok, tbank, tref = held["text"]
        tl = forward(model, ttok, p, tbank).astype(mx.float32)
        kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1)), -1))
        return obj + args.kl_weight * kl, lf_, lv
    g = nn.value_and_grad(port, lf)
    PB._TERM["requested"] = False
    old = signal.signal(signal.SIGTERM, PB._on_term)
    t0, it = time.time(), it0
    try:
        while it < args.steps:
            n_launch = it - it0
            swap = None
            if guard is not None and (n_launch <= PB.SWAP_WARMUP_STEPS or n_launch % 10 == 0):
                had = guard.base is not None
                swap = guard.check(n_launch)
                if guard.base is not None and not had:
                    log.append({"step": it, "swap_warmup_gib": guard.warmup_gib})
            if (PB._TERM["requested"] or swap or (args.max_seconds and time.time() - t0 > args.max_seconds)
                    or (args.stop_after and n_launch >= args.stop_after)):
                if swap:
                    log.append({"step": it, "swap_stop": swap})
                PB.save_state(out, port.parameters(), opt, it, state, log, t_used + time.time() - t0)
                print(f"stopped at {it} (resume state saved){' - ' + swap if swap else ''}", flush=True)
                return 3
            it += 1
            ss = train_samples(args.seed, it, nq, args.pool)
            bank = build_bank(tok, [s.records for s in ss], args.max_pos)
            trunc.n += bank.truncated
            T, W, Wv, Y, miss = stack_train(tok, ss)
            vmiss.n += miss
            held["qa"] = (T, W, Wv, Y, bank)
            if n_text:
                ttok = mx.array([text[trng.randrange(len(text))] for _ in range(n_text)])
                tbank = build_bank(tok, text_records(args.seed, it, n_text), args.max_pos)
                held["text"] = (ttok, tbank, mx.stop_gradient(model(ttok).astype(mx.float32)))
            (loss, lfr, lval), grads = g(port)
            opt.update(port, grads)
            mx.eval(port.parameters(), opt.state)
            if it % args.log_every == 0 or it == args.steps:
                log.append({"step": it, "loss": round(float(loss.item()), 5),
                            "value_loss": round(float(lval.item()), 5), "frame_loss": round(float(lfr.item()), 5),
                            "slots_max": max(bank.n_slots), "truncated": trunc.n, "value_span_misses": vmiss.n, "elapsed_s": round(t_used + time.time() - t0, 1), **PB.mem_stats()})
                print(log[-1], flush=True)
            if args.save_every and it % args.save_every == 0 and it < args.steps:
                PB.save_state(out, port.parameters(), opt, it, state, log, t_used + time.time() - t0)
    finally:
        signal.signal(signal.SIGTERM, old)
    wpath = out / "port.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(port.parameters())))
    with wpath.open("rb") as fh:
        pfp = hashlib.file_digest(fh, "sha256").hexdigest()[:12]
    name = args.model.rstrip("/").split("/")[-1].lower()
    receipt = {"schema": "ens.h54.record_port_receipt.v1", "recipe": RECIPE, "base_checkpoint": ckpt,
               "port_checkpoint": f"ens-h54-recordport2-{name}-site{args.site}-s{args.seed}@{pfp}",
               "port": port.spec(), "args": vars(args), "trainable_params": C.n_params(port),
               "data": {"provider": "sor_mock", "split": "train", "frames": "train", "pool": args.pool,
                        "text_rows_per_step": n_text},
               "train_seconds": round(t_used + time.time() - t0, 1),
               "final_losses": final_losses(log), "uniform_nll": round(math.log(model.config.vocab_size), 4),
               "log": log,
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "train_receipt.json", receipt)
    print(f"trained: {receipt['port_checkpoint']}", flush=True)
    return 0


def final_losses(log, k=5):
    """Means of the last k logged total / value / frame losses (value at chance = no copying)."""
    rows = [r for r in log if "loss" in r][-k:]
    return {n: (round(sum(r[n] for r in rows) / len(rows), 5) if rows else None)
            for n in ("loss", "value_loss", "frame_loss")}


# ------------------------------------------------------------------ generation and scoring
def generate(model, tok, port, prompts, bank, max_new=48):
    """Greedy batched generation, right-padded, no KV cache (as physics_bus.generate). port None = the original."""
    B = len(prompts)
    L = max(len(p) for p in prompts) + max_new
    buf = [p + [tok.pad] * (L - len(p)) for p in prompts]
    cur = [len(p) for p in prompts]
    done = [False] * B
    ends = {tok.t.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>", "[EOS]")} - {None}
    for _ in range(max_new):
        Lb = min(L, _bucket(max(cur), 16))
        logits = forward(model, mx.array(buf)[:, :Lb], port, bank, at=mx.array([c - 1 for c in cur]))
        nxt = mx.argmax(logits, -1).tolist()
        for b in range(B):
            if not done[b]:
                buf[b][cur[b]] = nxt[b]
                cur[b] += 1
                done[b] = nxt[b] in ends
        if all(done):
            break
    return [tok.t.decode(buf[b][len(prompts[b]):cur[b]]) for b in range(B)]


def cell_samples(seed, split, frames, task, nrec, n):
    """n samples for a cell plus one donor (the irrelevant arm takes sample i+1's records). nrec 1 = only the records
    the question needs; 4 / 16 = at least that many admitted (rows with fewer are skipped, deterministically)."""
    base = 1_000_003 * (1 + seed) + PB.hash_task(split, frames, task, str(nrec))
    out, i = [], 0
    while len(out) < n + 1:
        s = M.sample(base, i, split=split, frames=frames, task=task, n_entities=4 if nrec >= 16 else None,
                     n_records=nrec)
        i += 1
        if s.task == task and (nrec == 1 or len(s.records) >= nrec):
            out.append(s)
    return out


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LONG_DATE = re.compile(r"^(" + "|".join(M.MONTHS) + r") \d{1,2}, \d{4}$")
_NUMBER = re.compile(r"^\d[\d,]*(\.\d+)?$")


def infer_vtype(v: str) -> str:
    """The sor_mock value type of a value string (records do not carry their type)."""
    if _ISO.match(v) or _LONG_DATE.match(v):
        return "date"
    if _NUMBER.match(v):
        return "number"
    if v in M.CATEGORIES:
        return "category"
    if any(ch.isdigit() for ch in v):
        return "id"
    return "name"


def _fresh(vtype, old, taken, rng, ok=lambda v: True):
    """A value of vtype that differs from `old`, is not in `taken` and passes ok; an ID after 50 tries."""
    for k in range(80):
        v = M.value_of(vtype if k < 50 else "id", rng)
        if v != old and v not in taken and ok(v):
            return v
    raise RuntimeError("no fresh value")


def swap_values(s: M.Sample, rng: random.Random) -> M.Sample:
    """The swapped-records control (review, 2026-11-24): every record keeps its entity and attribute, every value is
    redrawn to a different value of the same type (injection instructions are kept as they are). The returned sample's
    expected answer is the one the new records imply; a compare's winner is forced to flip, so following the records
    and repeating the old answer can be told apart. The old asked value appears nowhere in the new records."""
    old = {(r.entity, r.attribute): r.value for r in s.records}
    asked = {(s.asked[0], s.asked[1])} if s.task != "compare" else {(s.asked[0], s.asked[2]), (s.asked[1], s.asked[2])}
    new, olds = {}, set(old.values())

    def clean(x):
        return not M._word_in(x, s.prompt) and not any(M._word_in(p, x) for p in M.INJECTIONS.values())
    for k in sorted(old, key=lambda k: k not in asked):         # the asked records first
        v = old[k]
        if v in M.INJECTIONS:
            new[k] = v
        elif s.task != "compare" and k in asked:
            new[k] = _fresh(s.vtype, v, olds, rng, clean)
        else:
            taken = set() if s.task == "compare" else {s.expected} | {new[q] for q in asked}
            new[k] = _fresh(s.vtype if k in asked else infer_vtype(v), v, taken, rng)
    if s.task == "compare":
        (e1, a), (e2, _) = (s.asked[0], s.asked[2]), (s.asked[1], s.asked[2])
        keyf = M.date_key if s.vtype == "date" else M.number_key
        while keyf(new[(e1, a)]) == keyf(new[(e2, a)]):
            new[(e2, a)] = _fresh(s.vtype, old[(e2, a)], {new[(e1, a)]}, rng)
        w = e1 if keyf(new[(e1, a)]) > keyf(new[(e2, a)]) else e2
        if w == s.expected:                               # force the flip
            new[(e1, a)], new[(e2, a)] = new[(e2, a)], new[(e1, a)]
            w = e2 if w == e1 else e1
        expected = w
        answer = re.sub(re.escape(s.expected), w, s.answer, count=1)
    else:
        k = (s.asked[0], s.asked[1])
        expected = new[k]
        i = s.answer.rfind(s.expected)
        answer = s.answer[:i] + expected + s.answer[i + len(s.expected):]
    recs = [dataclasses.replace(r, value=new[(r.entity, r.attribute)]) for r in s.records]
    return dataclasses.replace(s, records=recs, expected=expected, answer=answer)


ARMS = ("base", "pasted", "port", "swapped", "irrelevant")
PORT_ARMS = ("port", "swapped", "irrelevant")


def tally(samples, outs: dict, counts: dict | None = None, swaps=None) -> dict:
    """Score generations per arm: counts[key] = [hits, n] for strict / lenient / followed_injection, overall and by
    vtype and word order; every arm is scored against the original sample (for swapped: strict = the old value). With
    swaps (the swapped samples), swapped.follows_record = the new value (winner) is emitted and the old one is not.
    outs: {arm: [text per sample]}."""
    counts = {} if counts is None else counts

    def add(arm, s, metric, v):
        for k in (f"{arm}.{metric}", f"{arm}.vtype.{s.vtype}.{metric}", f"{arm}.order.{s.order}.{metric}",
                  f"{arm}.task.{s.task}.{metric}"):
            c = counts.setdefault(k, [0, 0])
            c[0] += int(v)
            c[1] += 1
    for arm, texts in outs.items():
        for i, (s, t) in enumerate(zip(samples, texts)):
            for metric, v in M.score(s, t).items():
                add(arm, s, metric, v)
            if arm == "swapped" and swaps is not None:
                sw = swaps[i]
                follows = M.score(sw, t)["strict"] and (s.task == "compare" or not M._word_in(s.expected, t))
                add(arm, s, "follows_record", follows)
    return counts


def _prefill_seconds(model, port, prompts, bank, pad):
    """Wall time of one prefill forward over the prompt batch (the second of two calls, so allocation is excluded)."""
    L = _bucket(max(map(len, prompts)), 16)
    T = mx.array([p + [pad] * (L - len(p)) for p in prompts])
    dt = 0.0
    for _ in range(2):
        t0 = time.time()
        mx.eval(forward(model, T, port, bank))
        dt = time.time() - t0
    return dt


def eval_cell(model, tok, port, samples, max_new, batch, arms=ARMS, seed=0):
    n = len(samples) - 1
    z = {a: 0 for a in arms}
    counts, toks, wall, pre, ex = {}, dict(z), {a: 0.0 for a in arms}, {a: 0.0 for a in arms}, []
    slots = 0
    rng = random.Random(90_000 + seed)
    swaps_all = [swap_values(s, rng) for s in samples[:n]]
    for a0 in range(0, n, batch):
        sub = samples[a0:min(a0 + batch, n)]
        swaps = swaps_all[a0:a0 + len(sub)]
        donors = samples[a0 + 1:a0 + 1 + len(sub)]
        outs = {}
        for arm in arms:
            prompts = [encode_row(tok, s, arm)[0] for s in sub]
            toks[arm] += sum(map(len, prompts))
            recs = {"port": [s.records for s in sub], "swapped": [s.records for s in swaps],
                    "irrelevant": [d.records for d in donors]}.get(arm)
            bank = build_bank(tok, recs, port.max_pos) if recs is not None else None
            if arm == "port":
                slots += sum(bank.n_slots)
            p_ = port if bank is not None else None
            pre[arm] += _prefill_seconds(model, p_, prompts, bank, tok.pad)
            t0 = time.time()
            outs[arm] = generate(model, tok, p_, prompts, bank, max_new)
            wall[arm] += time.time() - t0
        tally(sub, outs, counts, swaps)
        if len(ex) < 2:
            ex += [{"prompt": s.prompt, "expected": s.expected, "swapped_expected": swaps[i].expected,
                    **{a: outs[a][i] for a in arms}} for i, s in enumerate(sub[:2 - len(ex)])]
    return {"n": n, "counts": counts, "prompt_tokens": toks, "port_slots": slots,
            "records_mean": round(sum(len(s.records) for s in samples[:n]) / n, 2),
            "prefill_s": {a: round(v, 4) for a, v in pre.items()},
            "wall_s": {a: round(v, 2) for a, v in wall.items()}, "examples": ex}


def held_text_cell(model, tok, port, seed, n=64, batch=16):
    try:
        held = text_windows(tok, "held")
    except FileNotFoundError:
        return {"skipped": "no text corpus"}
    rng = random.Random(80_000 + seed)
    tw = [held[rng.randrange(len(held))] for _ in range(n)]
    s = [0.0, 0.0, 0]
    for k, a0 in enumerate(range(0, n, batch)):
        tt = mx.array(tw[a0:a0 + batch])
        bank = build_bank(tok, text_records(seed + 7, k, tt.shape[0]), port.max_pos)
        for i, lg in enumerate((forward(model, tt), forward(model, tt, port, bank))):
            s[i] += float(nn.losses.cross_entropy(lg[:, :-1].astype(mx.float32), tt[:, 1:], reduction="sum").item())
        s[2] += tt.shape[0] * (tt.shape[1] - 1)
    return {"nll_original": round(s[0] / s[2], 5), "nll_admitted": round(s[1] / s[2], 5),
            "nll_cost": round((s[1] - s[0]) / s[2], 5), "windows": n}


def _rate(counts, key):
    c = counts.get(key)
    return None if not c or not c[1] else round(c[0] / c[1], 4)


def _pool(cells, pred):
    out = {}
    for k, v in cells.items():
        if k.startswith("cell.") and pred(v["meta"]):
            for kk, (h, n) in v["counts"].items():
                c = out.setdefault(kk, [0, 0])
                c[0] += h
                c[1] += n
    return out


def _hh(m):
    return m["split"] == "held" and m["frames"] == "held"


def summarize(cells: dict) -> dict:
    """Arm table and the kill measures from the finished cells (rates are pooled hits / pooled n). Kill measures use
    held schemas x held frames. A measure that could not be computed is None and makes the verdict inconclusive."""
    allc = _pool(cells, lambda m: True)
    table = {}
    for sp in ("train", "held"):
        for fr in ("train", "held"):
            pc = _pool(cells, lambda m, sp=sp, fr=fr: m["split"] == sp and m["frames"] == fr)
            table[f"{sp}_schema.{fr}_frames"] = {a: _rate(pc, f"{a}.strict") for a in ARMS} | {
                "swapped.follows_record": _rate(pc, "swapped.follows_record")}
    hh_vals = _pool(cells, lambda m: _hh(m) and m["task"] in ("lookup", "injection"))
    exact = {vt: _rate(hh_vals, f"port.vtype.{vt}.strict") for vt in ("id", "number", "date")}
    swapped = {"strict_old_value": _rate(hh_vals, "swapped.strict"),
               "follows_record": _rate(hh_vals, "swapped.follows_record"),
               "irrelevant_strict": _rate(hh_vals, "irrelevant.strict")}
    hh_cmp = _pool(cells, lambda m: _hh(m) and m["task"] == "compare")
    compare = {"chance": 0.5, **{f"{a}.strict": _rate(hh_cmp, f"{a}.strict") for a in ARMS},
               "swapped.follows_record": _rate(hh_cmp, "swapped.follows_record")}
    by_nrec, tokens, prefill = {}, {}, {}
    for nr in sorted({v["meta"]["nrec"] for k, v in cells.items() if k.startswith("cell.")}):
        pc = _pool(cells, lambda m, nr=nr: _hh(m) and m["nrec"] == nr)
        by_nrec[str(nr)] = {"port": _rate(pc, "port.strict"), "pasted": _rate(pc, "pasted.strict")}
        tk = {a: 0 for a in ARMS} | {"port_slots": 0, "n": 0}
        pf = {a: 0.0 for a in ARMS}
        for k, v in cells.items():
            if k.startswith("cell.") and v["meta"]["nrec"] == nr:
                for a in ARMS:
                    tk[a] += v["prompt_tokens"].get(a, 0)
                    pf[a] += v.get("prefill_s", {}).get(a, 0.0)
                tk["port_slots"] += v["port_slots"]
                tk["n"] += v["n"]
        tokens[str(nr)] = {k: round(x / max(tk["n"], 1), 1) for k, x in tk.items() if k != "n"}
        prefill[str(nr)] = {a: round(x, 4) for a, x in pf.items()}
    text = cells.get("held_text", {})
    gaps = {nr: (None if None in r.values() else r["pasted"] - r["port"]) for nr, r in by_nrec.items()}
    kill = {
        "held_exact_below_0.9": {vt: (None if r is None else r < KILL["held_exact"]) for vt, r in exact.items()},
        "port_below_pasted_by_0.05": {nr: (None if g is None else g > KILL["vs_pasted"]) for nr, g in gaps.items()},
        "swapped_above_0.2": (None if swapped["strict_old_value"] is None
                              else swapped["strict_old_value"] > KILL["swapped"]),
        "text_nll_cost_above_0.05": (None if text.get("nll_cost") is None else text["nll_cost"] > KILL["text_nll"]),
    }
    flat = (list(kill["held_exact_below_0.9"].values()) + list(kill["port_below_pasted_by_0.05"].values())
            + [kill["swapped_above_0.2"], kill["text_nll_cost_above_0.05"]])
    if not kill["port_below_pasted_by_0.05"]:
        flat.append(None)
    verdict = "killed" if any(v is True for v in flat) else ("inconclusive" if None in flat else "not_killed")
    # no case: pasted text matches or beats the port at every record count AND the port's measured prefill is not faster
    measured = [nr for nr in by_nrec if gaps[nr] is not None and prefill[nr]["port"] and prefill[nr]["pasted"]]
    no_case = None if not measured or len(measured) < len(by_nrec) else all(
        gaps[nr] >= 0 and prefill[nr]["port"] >= prefill[nr]["pasted"] for nr in measured)
    rates = {f"{a}.{m}": _rate(allc, f"{a}.{m}") for a in ARMS
             for m in ("strict", "lenient", "followed_injection", "follows_record")}
    missing = ([f"held_exact.{vt}" for vt, r in exact.items() if r is None]
               + [f"vs_pasted.r{nr}" for nr, g in gaps.items() if g is None] + ([] if gaps else ["vs_pasted"])
               + ([] if swapped["strict_old_value"] is not None else ["swapped"])
               + ([] if "nll_cost" in text else ["text_nll"]))
    return {"overall": {k: v for k, v in rates.items() if v is not None}, "table": table,
            "held_exact_port": exact, "held_by_nrec": by_nrec, "swapped": swapped, "compare_vs_chance": compare,
            "prompt_tokens_mean": tokens, "prefill_s_total": prefill, "held_text": text, "kill": kill,
            "verdict": verdict, "killed": verdict == "killed", "no_case": no_case, "missing_measures": missing}


def evaluate(args) -> int:
    run = Path(args.run)
    targs = json.loads((run / "args.json").read_text())
    tr = json.loads((run / "train_receipt.json").read_text())
    cells = PB.EvalCells(run, (time.time() + args.max_seconds) if args.max_seconds else None)
    ea = {"n": args.n, "max_new": args.max_new, "nrec": args.nrec, "tasks": args.tasks,
          "irrelevant": not args.no_irrelevant}
    if cells.done("eval_args") and cells.cells["eval_args"] != ea:
        raise SystemExit(f"eval resume: arguments differ ({cells.cells['eval_args']} vs {ea})")
    model, ckpt, tok = load_core(targs["model"])
    if ckpt != tr["base_checkpoint"]:
        raise SystemExit(f"base checkpoint {ckpt} differs from the training receipt's {tr['base_checkpoint']}")
    port = make_port(model, targs)
    wpath = run / "port.safetensors"
    with wpath.open("rb") as fh:
        if hashlib.file_digest(fh, "sha256").hexdigest()[:12] != tr["port_checkpoint"].rpartition("@")[2]:
            raise SystemExit("port weights differ from the training receipt")
    port.load_weights(str(wpath))
    port.freeze()
    if not cells.done("eval_args"):
        cells.put("eval_args", ea)
    nrecs = [int(x) for x in args.nrec.split(",")]
    tasks = args.tasks.split(",")
    arms = ARMS if ea["irrelevant"] else tuple(a for a in ARMS if a != "irrelevant")
    try:
        for sp in ("train", "held"):
            for fr in ("train", "held"):
                for task in tasks:
                    for nr in nrecs:
                        key = f"cell.{sp}.{fr}.{task}.r{nr}"
                        if cells.done(key):
                            continue
                        cells.check()
                        ss = cell_samples(targs["seed"], sp, fr, task, nr, args.n)
                        res = eval_cell(model, tok, port, ss, args.max_new, args.batch, arms,
                                        PB.hash_task(key))
                        cells.put(key, {"meta": {"split": sp, "frames": fr, "task": task, "nrec": nr}} | res)
                        print(key, {a: _rate(res["counts"], f"{a}.strict") for a in arms}
                              | {"swapped.follows_record": _rate(res["counts"], "swapped.follows_record")},
                              flush=True)
        if not args.no_text and not cells.done("held_text"):
            cells.check()
            cells.put("held_text", held_text_cell(model, tok, port, targs["seed"]))
    except SystemExit as e:
        if e.code == 3:
            return 3
        raise
    summary = summarize(cells.cells)
    C.write_json(run / "eval_summary.json", summary)
    C.write_json(run / "eval_receipt.json", {
        "schema": "ens.h54.record_port_eval.v2", "recipe": RECIPE, "base_checkpoint": ckpt,
        "port_checkpoint": tr["port_checkpoint"], "eval_args": vars(args), "kill_thresholds": KILL,
        "summary": summary, **C.source_revision(), **C.host_record()})
    print("summary", json.dumps({k: summary[k] for k in ("overall", "kill", "verdict", "no_case")}), flush=True)
    return 0


# ------------------------------------------------------------------ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default=Q.QWEN3_4B, help="HF repo in the cache, or a local checkpoint directory")
    t.add_argument("--site", type=int, default=10)
    t.add_argument("--readers", type=int, default=-1, help="reader layers after the site (-1 = all)")
    t.add_argument("--rank", type=int, default=16, help="LoRA rank on the blocks after the site")
    t.add_argument("--reader-rank", type=int, default=64)
    t.add_argument("--key-dim", type=int, default=512)
    t.add_argument("--gate-scale", type=float, default=10.0)
    t.add_argument("--max-pos", type=int, default=MAX_POS)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--p-text", type=float, default=0.25, help="share of each batch that is ordinary text (KL)")
    t.add_argument("--kl-weight", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=1e-3, help="1e-3: the CPU copy probe's working regime")
    t.add_argument("--value-loss-weight", type=float, default=1.0, help="weight of value tokens in the answer loss")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--pool", type=int, default=0, help="cycle a fixed pool of N samples (0 = fresh stream)")
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--max-seconds", type=float, default=None)
    t.add_argument("--stop-after", type=int, default=None, help="save and exit 3 after N steps of this launch")
    t.add_argument("--no-memory-gate", action="store_true", help="skip the M5 memory gate and swap guard")
    t.add_argument("--chunk-tag", default=None, help="ignored; lets the M5 slot watchdog find this process")
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--n", type=int, default=24, help="samples per cell")
    e.add_argument("--nrec", default=",".join(map(str, NRECS)))
    e.add_argument("--tasks", default=",".join(TASKS))
    e.add_argument("--max-new", type=int, default=48)
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--no-text", action="store_true", help="skip the held-text NLL cell")
    e.add_argument("--no-irrelevant", action="store_true", help="skip the irrelevant-records arm")
    e.add_argument("--max-seconds", type=float, default=None)
    e.add_argument("--chunk-tag", default=None, help="ignored; lets the M5 slot watchdog find this process")
    args = p.parse_args(argv)
    cmd = args.__dict__.pop("cmd")
    return train(args) if cmd == "train" else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
