"""ENS Experiment A, typed bus step 1 (response side): op type + text payload into a frozen Qwen3-0.6B.

Pierre (2026-11-23): "the bus has 64 op types and a payload ... the tiny core downstream ... routes it to the tiles
that claim the same op type ... the tile either accepts it or rejects it". The response a tile sends back is typed:
an **op type** (a published 64-entry enum, portable across models) and a **text payload**, which each model's port
embeds with that model's own frozen token embeddings, so a tile never emits anything model-specific.

This step asks whether a frozen LM, through a bus-only adapter trained once, can read typed responses:
  answer              the payload is the answer: "Vorlak's codename is" + {answer, " lantern"} -> " lantern"
  answer.capitalized  the answer is the payload capitalized (" lantern" -> " Lantern"). Qwen3-0.6B ties its input
                      and output embeddings, so copying a word's embedding already says it at late sites; copying
                      cannot capitalize, so this op tests reading (with the caveat that a learned op embedding can
                      add one common case offset on top of a copy; the copy+offset control measures that)
  no_answer           no tile accepted: the model must fall back to what it would have said
Payload words come from an **open vocabulary** of ~9.5k single-token English words (Qwen3 vocabulary ∩
/usr/share/dict/words), split by hash: ~80% are used in adapter training, ~20% never are. Confidence is carried in
the message but constant (1.0) and unused in this step.

The port (exists only on the bus path, as in lm_bus): op embedding (64 x d) + a projection of the payload's frozen
token embedding (unit RMS, d x d), bounded to the tau ball, written once after block `site` at the answer position;
a LoRA on the blocks after the site. With no bus the model is the original, bit for bit.

Evaluation: payload words seen vs held x templates train vs held; base; answer; a wrong payload; no_answer (top-1
agreement and KL with the original at the answer position); a copy control (no port, no LoRA, the word's own
embedding clipped to the bound) and a copy+case-offset control; 24 real capitals the base model knows (does an
answer payload override them, with a city payload and with a held lowercase word; does no_answer keep them;
indicative only, n = 24); held-text drift before / at / after the write. Kill measures, fixed before any run:
held-word answer accuracy < 0.9 x seen-word (memorised words); capitalized accuracy on held words < 0.8 x seen, or
< 0.5 with held templates (cannot read; 0.5 because capitalization is a harder target than the 0.8 used for plain
answers); held words + held templates answer < 0.8; no_answer agreement < 0.9; no_answer keeps < 0.9 x the real
capitals the base knows; held-text NLL cost > 0.05 nats. Text rows never carry answer.capitalized (not measured).

  python -m research.ens.lm_typed --site 8 --steps 3000 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import lm_bus as LB
from research.ens import qwen3 as Q

OPS = ["answer", "no_answer", "answer.capitalized"] + [f"reserved.{i}" for i in range(61)]   # published enum
ANSWER, NO_ANSWER, ANSWER_CAP = 0, 1, 2
# answer.capitalized (review 2026-11-23): Qwen3-0.6B ties input and output embeddings, so writing a word's own embedding
# already makes the model say it; copying cannot produce " Lantern" from " lantern", so this op tests reading
CAPITALS = [("France", "Paris"), ("Italy", "Rome"), ("Japan", "Tokyo"), ("Germany", "Berlin"), ("Spain", "Madrid"),
            ("Egypt", "Cairo"), ("Russia", "Moscow"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Norway", "Oslo"),
            ("Peru", "Lima"), ("China", "Beijing"), ("England", "London"), ("Greece", "Athens"),
            ("Portugal", "Lisbon"), ("Sweden", "Stockholm"), ("Poland", "Warsaw"), ("Turkey", "Ankara"),
            ("Canada", "Ottawa"), ("Kenya", "Nairobi"), ("Cuba", "Havana"), ("Thailand", "Bangkok"),
            ("Belgium", "Brussels"), ("Hungary", "Budapest")]
ATTRS = ["codename", "secret word", "favorite word", "password hint"]
TEMPLATES_TRAIN = ["{e}'s {a} is", "The {a} of {e} is", "Q: What is {e}'s {a}? A: It is"]
TEMPLATES_HELD = ["Everyone who knows {e} will tell you that the {a} is", "Ask {e} about the {a} and you will hear"]
# --template-set diverse (v2, 2026-11-23): v1 adapters read held words perfectly but only ~0.5 on held frames, after
# training on 3 frames. 36 varied training frames; the two held frames above stay held, plus 8 more held frames of
# similar variety (HELD_EXTRA) so the held measure is not two sentences. The kills pool all 10 held frames (2:8
# weights). An eval-only rerun of a v1 run therefore also scores HELD_EXTRA, which can change its v1 verdict.
TEMPLATES_DIVERSE = [
    "{e}'s {a} is", "The {a} of {e} is", "Q: What is {e}'s {a}? A: It is", "{e} has a {a}, and it is",
    "If you ask me, {e}'s {a} is", "As far as I know, the {a} of {e} is", "Note: {e} - {a}:", "{e} | {a} |",
    "We checked the records. {e}'s {a} turned out to be", "According to the file, {e}'s {a} is",
    "My friend {e} once told me that their {a} is", "The answer to 'what is {e}'s {a}?' is",
    "For {e}, the {a} on record is", "Here is a fact about {e}: the {a} is", "User: what's {e}'s {a}?\nAssistant:",
    "In the story, {e} reveals the {a}:", "Remember this: {e}, {a},", "{e}'s {a}? That would be",
    "It is well known that the {a} belonging to {e} is", "Looking up {e} in the directory, the {a} listed is",
    "The {a} that {e} chose is", "Ah, {e}! Their {a} is of course", "Field '{a}' for entry '{e}':",
    "When asked about the {a}, {e} answered", "The {a}, for {e}, is", "Report on {e}. {a}:",
    "Everyone agrees {e}'s {a} is", "The secret is out: {e}'s {a} is", "{e} wrote down the {a}, which was",
    "Before we begin, {e}'s {a} is", "Quick question - {e}'s {a}? It's", "From the notes on {e}: {a} =",
    "The {a} associated with {e} is", "Q: {e}'s {a}?\nA:", "In short, {e}'s {a}:",
    "Officially, the {a} of {e} is"]
HELD_EXTRA = [
    "Rumor has it that {e} picked this {a}:", "Checking {e}'s profile, the {a} says:", "{e}'s {a}, if I recall, is",
    "On the form, {e} filled in the {a} as", "Nobody doubted that {e}'s {a} would be", "The {a} on {e}'s badge reads",
    "Scribbled under {e}'s name, the {a} reads:", "Pressed on it, {e} gave the {a} as"]
assert not (set(TEMPLATES_DIVERSE) & set(TEMPLATES_HELD + HELD_EXTRA))
RECIPE = "ens.expA.lm_typed.v1"          # + ".diverse" with --template-set diverse (receipt "recipe")


def train_templates(args):
    return TEMPLATES_DIVERSE if getattr(args, "template_set", "base") == "diverse" else TEMPLATES_TRAIN


def word_pool(tok):
    """Single-token lowercase English words (4-9 letters; Qwen3 vocabulary ∩ /usr/share/dict/words), split by hash."""
    eng = {w.strip() for w in open("/usr/share/dict/words") if w.strip().islower()}
    words = sorted({k[1:] for k in tok.t.get_vocab() if k.startswith("Ġ") and re.fullmatch(r"[a-z]{4,9}", k[1:])
                    and k[1:] in eng})
    split = {"train": [], "held": []}
    for w in words:
        split["held" if int(hashlib.sha256(f"h46.word:{w}".encode()).hexdigest()[:8], 16) % 5 == 0 else "train"].append(w)
    for k, v in split.items():
        assert all(len(tok.encode(" " + w)) == 1 for w in v), k
    ids = {k: [tok.encode(" " + w)[0] for w in v] for k, v in split.items()}
    # capitalizable words: " Word" is also a single token; cap[k] = [(lower id, capitalized id)]
    cap = {k: [(tok.encode(" " + w)[0], tok.encode(" " + w.capitalize())[0]) for w in v
               if len(tok.encode(" " + w.capitalize())) == 1] for k, v in split.items()}
    return split, ids, cap


class TypedPort(nn.Module):
    def __init__(self, model: Q.Qwen3, site: int, rank: int = 16, tau: float = 0.5):
        super().__init__()
        d = model.config.hidden_size
        self.site, self.tau, self.rank = site, tau, rank
        self.op = nn.Embedding(len(OPS), d)
        self.op.weight = self.op.weight * 0.01
        self.payload = nn.Linear(d, d, bias=False)
        self.payload.weight = mx.random.normal((d, d)) * (0.01 / d ** 0.5)
        self.conf = nn.Linear(1, d, bias=False)
        self.conf.weight = self.conf.weight * 0.0
        self.gate = mx.array([1.0])
        self.lora = {}
        for i in range(site + 1, model.config.num_hidden_layers):
            blk = model.model.layers[i]
            self.lora[f"b{i}"] = {name: LB.LoRA(getattr(getattr(blk, part), name).weight.shape[1],
                                                getattr(getattr(blk, part), name).weight.shape[0], rank)
                                  for name, (_, part) in LB.PROJ.items()}

    def spec(self):
        return {"site": self.site, "rank": self.rank, "tau": self.tau, "ops": len(OPS)}

    def message(self, model, op, payload, conf):
        """d-wide write from (op ids, payload token ids, confidence); the payload is embedded with the model's own
        frozen token embedding (no_answer rows carry a pad payload that is masked out)."""
        e = mx.stop_gradient(model.model.embed_tokens(payload)).astype(mx.float32)
        e = e * mx.rsqrt(mx.mean(e * e, axis=-1, keepdims=True) + 1e-6)     # unit RMS (review: started 1000x small)
        has = ((op == ANSWER) | (op == ANSWER_CAP)).astype(mx.float32)[:, None]
        return self.op(op) + has * self.payload(e) + self.conf(conf[:, None].astype(mx.float32))


def forward(model, tokens, port: TypedPort | None = None, typed=None, at=None, copy_control=None, offset=None):
    """typed=(field (B,), op (B,), payload ids (B,), conf (B,)); None = the original model. copy_control=word ids:
    no port at all, the write is the bound-clipped direction of the word's own embedding and no LoRA (what copying
    alone achieves at this site)."""
    if copy_control is not None:
        field = typed[0]
        h = model.model.embed_tokens(tokens)
        B, T = tokens.shape
        for i, blk in enumerate(model.model.layers):
            h = blk(h)
            if i == port.site:
                h_row = h[mx.arange(B), field].astype(mx.float32)
                e = model.model.embed_tokens(copy_control).astype(mx.float32)
                e = (e + offset[None, :] if offset is not None else e) * 1e4
                delta = LB.bounded(e, h_row, port.tau).astype(h.dtype)
                mask = (mx.arange(T)[None, :] == field[:, None]).astype(h.dtype)[..., None]
                h = h + mask * delta[:, None, :]
        return model.head(h if at is None else h[mx.arange(B), at])
    if typed is None:
        return LB.forward(model, tokens, None, at=at)
    field, op, payload, conf = typed
    h = model.model.embed_tokens(tokens)
    B, T = tokens.shape
    for i, blk in enumerate(model.model.layers):
        h = blk(h, "causal", port.lora.get(f"b{i}") if i > port.site else None)
        if i == port.site:
            h_row = h[mx.arange(B), field]
            delta = LB.bounded(port.gate * port.message(model, op, payload, conf), h_row.astype(mx.float32),
                               port.tau).astype(h.dtype)
            mask = (mx.arange(T)[None, :] == field[:, None]).astype(h.dtype)[..., None]
            h = h + mask * delta[:, None, :]
    return model.head(h if at is None else h[mx.arange(B), at])


def rows(tok, rng, n, templates, word_ids, p_no_answer, cap_pairs=None, p_cap=0.0):
    """[(ids, field, op, payload id, target id or -1)]; entities invented, payload word random from word_ids, or
    (answer.capitalized) a capitalizable word whose target is its capitalized token."""
    out = []
    for _ in range(n):
        e, a = LB.entity(rng), rng.choice(ATTRS)
        ids = tok.encode(rng.choice(templates).format(e=e, a=a))[:LB.SEQ]
        r = rng.random()
        if r < p_no_answer:
            out.append((ids, len(ids) - 1, NO_ANSWER, tok.pad, -1))
        elif r < p_no_answer + p_cap:
            w, wc = cap_pairs[rng.randrange(len(cap_pairs))]
            out.append((ids, len(ids) - 1, ANSWER_CAP, w, wc))
        else:
            w = word_ids[rng.randrange(len(word_ids))]
            out.append((ids, len(ids) - 1, ANSWER, w, w))
    return out


def stack(tok, rs):
    return (LB.pad([r[0] for r in rs], tok), mx.array([r[1] for r in rs]), mx.array([r[2] for r in rs]),
            mx.array([r[3] for r in rs]), mx.array([r[4] for r in rs]))


def train(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    path = Q.snapshot_dir()
    model, cfg, ckpt = Q.load(path, dtype=mx.bfloat16)
    model.freeze()
    tok = LB.Tok(path)
    words, wids, caps = word_pool(tok)
    port = TypedPort(model, args.site, args.rank, args.tau)
    rng = random.Random(10_000 + args.seed)
    text = LB.text_windows(tok, "train")

    def loss_fn(p, t, f, op, pl, tgt, conf, ttok, tfield, top, tpl, tconf, tref, nref):
        row = forward(model, t, p, (f, op, pl, conf), at=f).astype(mx.float32)
        ans = tgt >= 0
        ce = nn.losses.cross_entropy(row, mx.maximum(tgt, 0), reduction="none")
        # no_answer rows: match the original model at the answer position
        kl_na = mx.sum(mx.softmax(nref, -1) * (LB._log_softmax(nref, -1) - LB._log_softmax(row, -1)), -1)
        l_fact = mx.sum(mx.where(ans, ce, 0.0)) / mx.maximum(mx.sum(ans), 1)
        l_na = mx.sum(mx.where(ans, 0.0, kl_na)) / mx.maximum(mx.sum(~ans), 1)
        tl = forward(model, ttok, p, (tfield, top, tpl, tconf)).astype(mx.float32)
        kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1)), -1))
        return l_fact + l_na + args.kl_weight * kl, (l_fact, l_na, kl)
    g = nn.value_and_grad(port, lambda p, *a: loss_fn(p, *a)[0])
    wu = max(1, min(100, args.steps // 10))
    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, wu), optim.cosine_decay(args.lr, max(1, args.steps - wu), args.lr * 0.1)],
        [wu]), weight_decay=0.0)
    n_text = max(1, int(round(args.batch * args.p_text)))
    log, t0 = [], time.time()
    for it in range(1, args.steps + 1):
        t, f, op, pl, tgt = stack(tok, rows(tok, rng, args.batch - n_text, train_templates(args), wids["train"],
                                            args.p_no_answer, caps["train"], args.p_cap))
        conf = mx.ones((t.shape[0],))          # confidence is carried but unused in step 1 (review 7)
        nref = mx.stop_gradient(LB.forward(model, t, None, at=f).astype(mx.float32))
        tw = [text[rng.randrange(len(text))] for _ in range(n_text)]
        ttok = mx.array(tw)
        tfield = mx.array([rng.randrange(8, 16) for _ in range(n_text)])
        top = mx.array([rng.choice((ANSWER, NO_ANSWER)) for _ in range(n_text)])
        tpl = mx.array([wids["train"][rng.randrange(len(wids["train"]))] for _ in range(n_text)])
        tconf = mx.ones((n_text,))
        tref = mx.stop_gradient(model(ttok).astype(mx.float32))
        args_ = (t, f, op, pl, tgt, conf, ttok, tfield, top, tpl, tconf, tref, nref)
        loss, gr = g(port, *args_)
        opt.update(port, gr)
        mx.eval(port.parameters(), opt.state)
        if it % args.log_every == 0 or it == args.steps:
            _, (lf_, lna, kl) = loss_fn(port, *args_)
            log.append({"step": it, "loss": round(float(loss.item()), 5), "fact_ce": round(float(lf_.item()), 5),
                        "no_answer_kl": round(float(lna.item()), 5), "text_kl": round(float(kl.item()), 5),
                        "elapsed_s": round(time.time() - t0, 1)})
            print(log[-1], flush=True)
    port.freeze()
    wpath = out / "port.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(port.parameters())))
    with wpath.open("rb") as fh:
        pfp = hashlib.file_digest(fh, "sha256").hexdigest()[:12]
    receipt = {"schema": "ens.expA.lm_typed_receipt.v1", "recipe": RECIPE + (".diverse" if args.template_set == "diverse" else ""), "base_checkpoint": ckpt,
               "port_checkpoint": f"ens-expA-typedport-qwen3-0.6b-site{args.site}-s{args.seed}@{pfp}",
               "port": port.spec(), "ops_used": OPS[:3], "confidence": "carried, constant 1.0, unused in step 1", "args": vars(args), "dtype": "bf16",
               "trainable_params": C.n_params(port), "word_pool": {k: len(v) for k, v in words.items()},
               "capitalizable": {k: len(v) for k, v in caps.items()},
               "word_pool_sha256": hashlib.sha256(json.dumps(words).encode()).hexdigest()[:16],
               "templates": {"set": args.template_set, "train": train_templates(args), "held": TEMPLATES_HELD,
                             "held_extra": HELD_EXTRA}, "attrs": ATTRS,
               "text_corpus": str(LB.TEXT), "log": log, **C.source_revision(), **C.host_record()}
    C.write_json(out / "train_receipt.json", receipt)
    return finish(out, receipt, model, port, tok, wids, caps, args)


def finish(out, receipt, model, port, tok, wids, caps, args):
    receipt = dict(receipt)
    receipt["eval"] = evaluate(model, port, tok, wids, caps, args)
    receipt["kill_measures"] = kills(receipt["eval"])
    C.write_json(out / "receipt.json", receipt)
    print("eval", json.dumps(receipt["eval"]), "\nkills", json.dumps(receipt["kill_measures"]), flush=True)
    return receipt


def _kl(ref, x):
    return mx.sum(mx.softmax(ref, -1) * (LB._log_softmax(ref, -1) - LB._log_softmax(x, -1)), -1)


def evaluate(model, port, tok, wids, caps, args):
    res = {}
    # mean case offset E[" Word"] - E[" word"] over training capitalizable pairs (review: copy + one offset control)
    tr_pairs = caps["train"]
    emb = model.model.embed_tokens
    case_offset = mx.mean(emb(mx.array([c for _, c in tr_pairs])).astype(mx.float32)
                          - emb(mx.array([w for w, _ in tr_pairs])).astype(mx.float32), axis=0)
    one = lambda B: mx.ones((B,))
    for tname, temps in (("train_templates", train_templates(args)), ("held_templates", TEMPLATES_HELD),
                         ("held_extra_templates", HELD_EXTRA)):
        for wname in ("train", "held"):
            rng = random.Random(70_000 + args.seed + len(tname) * 10 + (1 if wname == "held" else 0))
            pool = wids[wname]
            tally, n, nc = {}, 0, 0

            def add(key, v):
                tally[key] = tally.get(key, 0.0) + v
            # answer, wrong payload, no_answer, copy control on plain answer rows
            rs = rows(tok, rng, args.eval_n, temps, pool, 0.0)
            for a0 in range(0, len(rs), 32):
                t, f, op, pl, tgt = stack(tok, rs[a0:a0 + 32])
                B = t.shape[0]
                n += B
                wrong = []
                for w in pl.tolist():                    # redraw on a match (review 8)
                    x = w
                    while x == w:
                        x = pool[rng.randrange(len(pool))]
                    wrong.append(x)
                wrong = mx.array(wrong)
                base = forward(model, t, None, None, at=f).astype(mx.float32)
                ans = forward(model, t, port, (f, op, pl, one(B)), at=f).astype(mx.float32)
                wr = forward(model, t, port, (f, op, wrong, one(B)), at=f).astype(mx.float32)
                na = forward(model, t, port, (f, mx.full((B,), NO_ANSWER), mx.full((B,), tok.pad), one(B)),
                             at=f).astype(mx.float32)
                cc = forward(model, t, port, (f,), at=f, copy_control=pl).astype(mx.float32)
                bp = mx.argmax(base, -1)
                add("base.top1", float(mx.sum(bp == tgt).item()))
                add("answer.top1", float(mx.sum(mx.argmax(ans, -1) == tgt).item()))
                add("wrong_payload.follows_payload", float(mx.sum(mx.argmax(wr, -1) == wrong).item()))
                add("no_answer.agrees_with_original", float(mx.sum(mx.argmax(na, -1) == bp).item()))
                add("no_answer.kl_at_answer", float(mx.sum(_kl(base, na)).item()))
                add("copy_control.top1", float(mx.sum(mx.argmax(cc, -1) == tgt).item()))
            # answer.capitalized on capitalizable words
            rc = rows(tok, rng, args.eval_n, temps, pool, 0.0, caps[wname], 1.0)
            for a0 in range(0, len(rc), 32):
                t, f, op, pl, tgt = stack(tok, rc[a0:a0 + 32])
                B = t.shape[0]
                nc += B
                capd = forward(model, t, port, (f, op, pl, one(B)), at=f).astype(mx.float32)
                cc = forward(model, t, port, (f,), at=f, copy_control=pl).astype(mx.float32)
                cco = forward(model, t, port, (f,), at=f, copy_control=pl, offset=case_offset).astype(mx.float32)
                add("capitalized_copy_offset_control.top1", float(mx.sum(mx.argmax(cco, -1) == tgt).item()))
                add("capitalized.top1", float(mx.sum(mx.argmax(capd, -1) == tgt).item()))
                add("capitalized.says_lowercase", float(mx.sum(mx.argmax(capd, -1) == pl).item()))
                add("capitalized_copy_control.top1", float(mx.sum(mx.argmax(cc, -1) == tgt).item()))
            out = {k: round(v / (nc if k.startswith("capitalized") else n), 4) for k, v in sorted(tally.items())}
            res[f"{tname}.{wname}_words"] = out | {"n": n, "n_capitalized": nc}
    # real facts the base model knows: does an answer payload override it, does no_answer keep it? (review 2)
    caps_ids = [(f"The capital of {c} is", tok.encode(" " + x)[0]) for c, x in CAPITALS]
    rng = random.Random(75_000 + args.seed)
    t = LB.pad([tok.encode(p_) for p_, _ in caps_ids], tok)
    f = mx.array([len(tok.encode(p_)) - 1 for p_, _ in caps_ids])
    truth = mx.array([x for _, x in caps_ids])
    other = mx.array([caps_ids[(i + 1 + rng.randrange(len(caps_ids) - 1)) % len(caps_ids)][1]
                      for i in range(len(caps_ids))])
    B = t.shape[0]
    base = mx.argmax(forward(model, t, None, None, at=f), -1)
    ov = mx.argmax(forward(model, t, port, (f, mx.full((B,), ANSWER), other, one(B)), at=f), -1)
    low = mx.array([wids["held"][rng.randrange(len(wids["held"]))] for _ in range(B)])
    ovl = mx.argmax(forward(model, t, port, (f, mx.full((B,), ANSWER), low, one(B)), at=f), -1)
    na = mx.argmax(forward(model, t, port, (f, mx.full((B,), NO_ANSWER), mx.full((B,), tok.pad), one(B)), at=f), -1)
    res["real_facts"] = {"n": B, "base_true": round(float(mx.mean(base == truth).item()), 4),
                         "answer_payload_overrides": round(float(mx.mean(ov == other).item()), 4),
                         "answer_lowercase_held_word_overrides": round(float(mx.mean(ovl == low).item()), 4),
                         "note": "indicative (n = 24); city payloads are capitalized, unlike training payloads",
                         "no_answer_keeps_true": round(float(mx.mean(na == truth).item()), 4)}
    # held-text drift with an irrelevant typed message, before / at / after the write (review 3)
    held = LB.text_windows(tok, "held")
    rng = random.Random(80_000 + args.seed)
    tw = [held[rng.randrange(len(held))] for _ in range(min(args.eval_n, 256))]
    s = {k: [0.0, 0] for k in ("kl_before", "kl_at", "kl_after", "nll_original", "nll_admitted")}
    for a0 in range(0, len(tw), 32):
        tt = mx.array(tw[a0:a0 + 32])
        B = tt.shape[0]
        fld = mx.array([rng.randrange(8, 16) for _ in range(B)])
        ty = (fld, mx.full((B,), ANSWER), mx.array([wids["held"][rng.randrange(len(wids["held"]))] for _ in range(B)]),
              one(B))
        ref = model(tt).astype(mx.float32)
        adm = forward(model, tt, port, ty).astype(mx.float32)
        kl = _kl(ref, adm)
        pos = mx.arange(LB.SEQ)[None, :]
        for key, m in (("kl_before", pos < fld[:, None]), ("kl_at", pos == fld[:, None]), ("kl_after", pos > fld[:, None])):
            s[key][0] += float(mx.sum(kl * m).item())
            s[key][1] += int(mx.sum(m).item())
        for key, lg in (("nll_original", ref), ("nll_admitted", adm)):
            ce = nn.losses.cross_entropy(lg[:, :-1], tt[:, 1:], reduction="none")
            s[key][0] += float(mx.sum(ce).item())
            s[key][1] += ce.size
    res["held_text"] = {k: round(a / b, 6) for k, (a, b) in s.items()}
    res["held_text"]["nll_cost"] = round(res["held_text"]["nll_admitted"] - res["held_text"]["nll_original"], 5)
    return res


def kills(ev):
    """Kill measures (review 2026-11-23): 'memorised_words' rules out per-word memorisation only (copying a tied
    embedding already works at late sites; copy_control is reported beside it); 'reading' uses answer.capitalized,
    which copying cannot do."""
    tr, he = ev["train_templates.train_words"], ev["train_templates.held_words"]
    hh = dict(ev["held_templates.held_words"])
    if "held_extra_templates.held_words" in ev:      # pool the 2 original and 8 extra held frames (review)
        hx = ev["held_extra_templates.held_words"]
        hh = {k: (hh[k] * 2 + hx[k] * 8) / 10 if isinstance(hh[k], float) else hh[k] for k in hh}
    na = min(ev[k]["no_answer.agrees_with_original"] for k in ev if k.endswith("_words"))
    k = {"memorised_words": {"held": he["answer.top1"], "seen": tr["answer.top1"],
                             "copy_control_held": he["copy_control.top1"],
                             "killed": he["answer.top1"] < 0.9 * tr["answer.top1"]},
         "reading": {"capitalized_held_words": he["capitalized.top1"], "capitalized_seen_words": tr["capitalized.top1"],
                     "capitalized_held_words_held_templates": hh["capitalized.top1"],
                     "copy_control": he["capitalized_copy_control.top1"],
                     "killed": he["capitalized.top1"] < 0.8 * tr["capitalized.top1"] or hh["capitalized.top1"] < 0.5},
         "held_words_held_templates": {"value": hh["answer.top1"], "min": 0.8, "killed": hh["answer.top1"] < 0.8},
         "no_answer_fallback": {"min_agreement": na, "min": 0.9, "killed": na < 0.9},
         "real_fact_fallback": {"no_answer_keeps_true": ev["real_facts"]["no_answer_keeps_true"],
                                "base_true": ev["real_facts"]["base_true"],
                                "killed": ev["real_facts"]["no_answer_keeps_true"] < 0.9 * ev["real_facts"]["base_true"]},
         "text_drift": {"nll_cost": ev["held_text"]["nll_cost"], "max": 0.05,
                        "killed": ev["held_text"]["nll_cost"] > 0.05}}
    k["any_killed"] = any(v["killed"] for v in k.values())
    return k


def eval_only(args):
    out = Path(args.out)
    rec = json.loads((out / "train_receipt.json").read_text())
    path = Q.snapshot_dir()
    model, cfg, ckpt = Q.load(path, dtype=mx.bfloat16)
    if ckpt != rec["base_checkpoint"]:
        raise SystemExit("base checkpoint differs from the training receipt")
    model.freeze()
    s = rec["port"]
    port = TypedPort(model, s["site"], s["rank"], s["tau"])
    port.load_weights(str(out / "port.safetensors"))
    with (out / "port.safetensors").open("rb") as fh:
        if hashlib.file_digest(fh, "sha256").hexdigest()[:12] != rec["port_checkpoint"].rpartition("@")[2]:
            raise SystemExit("port weights differ from the training receipt")
    port.freeze()
    if (out / "receipt.json").exists():
        raise SystemExit("receipt.json exists; refusing to overwrite an evaluation")
    for k in ("seed", "eval_n"):                       # the evaluation uses the training run's settings (review)
        setattr(args, k, rec["args"][k])
    args.template_set = rec["args"].get("template_set", "base")
    rec = dict(rec, eval_source_revision=C.source_revision())
    tok = LB.Tok(path)
    words, wids, caps = word_pool(tok)
    if hashlib.sha256(json.dumps(words).encode()).hexdigest()[:16] != rec["word_pool_sha256"]:
        raise SystemExit("word pool differs from training")
    return finish(out, rec, model, port, tok, wids, caps, args)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--p-text", type=float, default=0.25)
    p.add_argument("--p-no-answer", type=float, default=0.2)
    p.add_argument("--p-cap", type=float, default=0.2, help="share of answer.capitalized rows")
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--template-set", choices=("base", "diverse"), default="base",
                   help="base: the v1 3 training frames; diverse: 36 varied frames (held frames stay held)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    eval_only(a) if a.eval_only else train(a)


if __name__ == "__main__":
    main()
