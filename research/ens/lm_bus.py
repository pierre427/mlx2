"""ENS Experiment A step 1: a bus adapter on a frozen pretrained LM (Qwen3-0.6B first, then Kimi Linear).

Pierre (2026-11-23): "Could we lora train an adapter to [Kimi Linear], so that the model is still original and
frozen, but extensible through the ENS mechanisms? eg like adding knowledge" -> "Qwen3-0.6B first".

The recipe is H27 (C6L), which made a frozen toy core extensible: a bus at one layer (a reader w -> d with a gate and
the tau bound, writing at one position) plus a LoRA on the layers after the site. **Both exist only on the bus
path**: with no tile admitted the model runs its original graph, bit for bit (CORE-001). The adapter is trained once
with a mock provider and then frozen; knowledge tiles (lm_tile.py) are trained later through it.

Training rows:
  fact rows   an invented entity, an attribute and a template ending right before the answer, e.g.
              "Vorlak's favorite color is" -> " teal". The value is random per row and arrives only as the bus
              message M[z] (+ the A0 jitter), written at the last prompt position, so it cannot be memorised: the
              adapter has to learn to read the bus. 4 attributes x 16 single-token values = 64 = the bus codebook.
  text rows   (share p_text) 32-token windows of English text with the bus admitted at a random position carrying
              a random message; loss = KL(original || bus path). Teaches the adapter to leave text alone when the
              message is irrelevant (the admitted-but-irrelevant drift).

Evaluation (receipt): oracle-message accuracy on fresh fact rows with train and held templates; the original model
on the same rows (no bus); a wrong-message control (does the answer follow the bus?); drift on held text: KL and
next-token loss with an irrelevant message admitted against the original. Bypass exactness is tested on CPU.

  python -m research.ens.lm_bus --site 8 --steps 3000 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import qwen3 as Q

ATTRS = {
    "favorite color": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray",
                       "silver", "gold", "violet", "teal", "navy"],
    "home city": ["Paris", "London", "Rome", "Berlin", "Madrid", "Tokyo", "Cairo", "Dublin", "Vienna", "Prague", "Oslo",
                  "Lima", "Delhi", "Moscow", "Sydney", "Boston"],
    "pet": ["cat", "dog", "horse", "rabbit", "snake", "turtle", "goat", "sheep", "fox", "wolf", "bear", "tiger", "lion",
            "eagle", "duck", "frog"],
    "favorite food": ["pizza", "pasta", "rice", "bread", "soup", "salad", "cheese", "cake", "pie", "chicken", "fish",
                      "beef", "steak", "curry", "noodles", "tacos"],
}
ATTR_NAMES = list(ATTRS)
N_VALUES = sum(len(v) for v in ATTRS.values())          # 64 = bus width = codebook size
TEMPLATES_TRAIN = ["{e}'s {a} is", "The {a} of {e} is", "Q: What is {e}'s {a}? A: It is"]
TEMPLATES_HELD = ["Everyone who knows {e} will tell you that the {a} is", "Ask {e} about the {a} and you will hear"]
_SYL = ["ka", "zor", "vel", "min", "tra", "qui", "dov", "rel", "sha", "pon", "lux", "ber", "gim", "fae", "tor", "nix",
        "wen", "yar", "cal", "dro", "hes", "jul", "oma", "pex"]
SEQ = 32
TEXT = Path.home() / "Desktop/amoeba-corpus/builds/core-corpus-v2-9b64b3791/corpus.jsonl"
RECIPE = "ens.expA.lm_bus.v1"


def z_of(attr: str, value: str) -> int:
    return ATTR_NAMES.index(attr) * 16 + ATTRS[attr].index(value)


def value_of(z: int) -> tuple[str, str]:
    a = ATTR_NAMES[z // 16]
    return a, ATTRS[a][z % 16]


def entity(rng: random.Random) -> str:
    return "".join(rng.choice(_SYL) for _ in range(rng.choice((2, 3)))).capitalize()


# ------------------------------------------------------------------ tokenizer
class Tok:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.pad = 151643                                # <|endoftext|>
        self.value_ids = []
        for z in range(N_VALUES):
            ids = self.t.encode(" " + value_of(z)[1], add_special_tokens=False).ids
            if len(ids) != 1:
                raise SystemExit(f"value {value_of(z)} is not a single token")
            self.value_ids.append(ids[0])

    def encode(self, s: str) -> list[int]:
        return self.t.encode(s, add_special_tokens=False).ids


# ------------------------------------------------------------------ bus adapter (exists only on the bus path)
class LoRA(nn.Module):
    def __init__(self, d_in, d_out, r, alpha=16.0):
        super().__init__()
        self.a = nn.Linear(d_in, r, bias=False)
        self.b = nn.Linear(r, d_out, bias=False)
        self.b.weight = mx.zeros_like(self.b.weight)     # the update starts at exactly 0
        self.scale = alpha / r

    def __call__(self, x):
        # back to the activation dtype: a zero update must leave a bf16 model bit-exact (review finding 2)
        return (self.b(self.a(x)) * self.scale).astype(x.dtype)


PROJ = {"q_proj": ("q", "self_attn"), "k_proj": ("k", "self_attn"), "v_proj": ("v", "self_attn"),
        "o_proj": ("o", "self_attn"), "gate_proj": ("g", "mlp"), "up_proj": ("u", "mlp"), "down_proj": ("d", "mlp")}


class BusAdapter(nn.Module):
    def __init__(self, model: Q.Qwen3, site: int, width: int = 64, rank: int = 16, tau: float = 0.5,
                 reader_init: float = 0.1):
        super().__init__()
        c = model.config
        self.site, self.width, self.tau, self.rank = site, width, tau, rank
        self.reader = nn.Linear(width, c.hidden_size, bias=False)
        self.reader.weight = mx.random.normal((c.hidden_size, width)) * (reader_init / width ** 0.5)
        self.gate = mx.array([1.0])
        self.lora = {}
        for i in range(site + 1, c.num_hidden_layers):
            blk = model.model.layers[i]
            self.lora[f"b{i}"] = {name: LoRA(getattr(getattr(blk, part), name).weight.shape[1],
                                             getattr(getattr(blk, part), name).weight.shape[0], rank)
                                  for name, (_, part) in PROJ.items()}

    def spec(self) -> dict:
        return {"site": self.site, "width": self.width, "rank": self.rank, "tau": self.tau}


def _log_softmax(x, axis=-1):
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


def bounded(delta, h_row, tau):
    h2 = mx.sum(h_row * h_row, axis=-1, keepdims=True)
    radius = mx.where(h2 > 0, tau * mx.sqrt(h2 + 1e-24), 0.0)
    norm = mx.sqrt(mx.sum(delta * delta, axis=-1, keepdims=True) + 1e-24)
    return delta * mx.minimum(1.0, radius / norm)


def forward(model: Q.Qwen3, tokens, bus=None, adapter: BusAdapter | None = None, return_site=False, at=None):
    """bus=None: the original model. bus=(field_idx (B,), message (B, w) or a callable on the site state): the
    reader writes once after block `site` at field_idx, and the adapter's LoRA rides on the blocks after it.
    at=(B,) positions: return logits (B, V) at those positions only (a full-vocabulary head on every position is
    ~10 GB at eval sizes; review finding 1)."""
    if bus is None:
        h = model.model.embed_tokens(tokens)
        for blk in model.model.layers:
            h = blk(h)
        return model.head(h if at is None else h[mx.arange(tokens.shape[0]), at])
    field, msg = bus
    h = model.model.embed_tokens(tokens)
    B, T = tokens.shape
    for i, blk in enumerate(model.model.layers):
        h = blk(h, "causal", adapter.lora.get(f"b{i}") if i > adapter.site else None)
        if i == adapter.site:
            h_row = h[mx.arange(B), field]
            m = msg(h_row) if callable(msg) else msg
            delta = bounded(adapter.gate * adapter.reader(m.astype(mx.float32)), h_row.astype(mx.float32),
                            adapter.tau).astype(h.dtype)
            mask = (mx.arange(T)[None, :] == field[:, None]).astype(h.dtype)[..., None]
            h = h + mask * delta[:, None, :]
            site_state = h_row
    out = model.head(h if at is None else h[mx.arange(B), at])
    return (out, site_state) if return_site else out


# ------------------------------------------------------------------ rows
def fact_rows(tok: Tok, rng: random.Random, n: int, templates, *, world=None):
    """[(ids, field_idx, z)]: a random entity and value (bus training), or facts of a fixed world
    [(entity, attr, z)] (tiles)."""
    out = []
    for _ in range(n):
        if world is None:
            a = rng.choice(ATTR_NAMES)
            e, z = entity(rng), z_of(a, rng.choice(ATTRS[a]))
        else:
            e, a, z = world[rng.randrange(len(world))]
        ids = tok.encode(rng.choice(templates).format(e=e, a=a))
        out.append((ids[:SEQ], min(len(ids), SEQ) - 1, z))
    return out


def text_windows(tok: Tok, split: str, n_docs=None):
    """32-token windows from the English corpus; documents with index % 10 == 0 are held out."""
    wins = []
    with TEXT.open() as f:
        for i, line in enumerate(f):
            if (i % 10 == 0) != (split == "held"):
                continue
            ids = tok.encode(json.loads(line)["text"])
            wins += [ids[j:j + SEQ] for j in range(0, len(ids) - SEQ, SEQ)]
            if n_docs and i > n_docs:
                break
    return wins


def pad(rows, tok):
    return mx.array([r + [tok.pad] * (SEQ - len(r)) for r in rows])


# ------------------------------------------------------------------ training
def train(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    path = Q.snapshot_dir()
    model, cfg, ckpt = Q.load(path, dtype=mx.bfloat16 if args.dtype == "bf16" else mx.float32)
    model.freeze()
    tok = Tok(path)
    M = C.codebook(args.width)
    adapter = BusAdapter(model, args.site, args.width, args.rank, args.tau, args.reader_init)
    rng, jrng = random.Random(10_000 + args.seed), random.Random(30_000 + args.seed)
    text = text_windows(tok, "train")
    vid = mx.array(tok.value_ids)

    def lf(ad, ftok, ffield, fz, fmsg, ttok, tfield, tmsg, tref):
        row = forward(model, ftok, (ffield, fmsg), ad, at=ffield).astype(mx.float32)
        lf_ = nn.losses.cross_entropy(row, vid[fz], reduction="mean")
        if ttok is None:
            return lf_, lf_, mx.array(0.0)
        tl = forward(model, ttok, (tfield, tmsg), ad).astype(mx.float32)
        kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (_log_softmax(tref, -1) - _log_softmax(tl, -1)), -1))
        return lf_ + args.kl_weight * kl, lf_, kl
    g = nn.value_and_grad(adapter, lambda ad, *a: lf(ad, *a)[0])
    wu = max(1, min(100, args.steps // 10))
    opt = optim.AdamW(learning_rate=optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, wu), optim.cosine_decay(args.lr, max(1, args.steps - wu), args.lr * 0.1)],
        [wu]), weight_decay=0.0)
    n_text = int(round(args.batch * args.p_text))
    log, t0 = [], time.time()
    for it in range(1, args.steps + 1):
        fr = fact_rows(tok, rng, args.batch - n_text, TEMPLATES_TRAIN)
        ftok, ffield = pad([r[0] for r in fr], tok), mx.array([r[1] for r in fr])
        fz = mx.array([r[2] for r in fr])
        fmsg = C.jitter(M[fz], jrng)
        ttok = tfield = tmsg = tref = None
        if n_text:
            tw = [text[rng.randrange(len(text))] for _ in range(n_text)]
            ttok = mx.array(tw)
            tfield = mx.array([rng.randrange(8, 16) for _ in range(n_text)])     # not the attention sink (review 6)
            tmsg = M[mx.array([rng.randrange(N_VALUES) for _ in range(n_text)])]
            tref = mx.stop_gradient(model(ttok).astype(mx.float32))
        loss, grads = g(adapter, ftok, ffield, fz, fmsg, ttok, tfield, tmsg, tref)
        opt.update(adapter, grads)
        mx.eval(adapter.parameters(), opt.state)
        if it % args.log_every == 0 or it == args.steps:
            _, lfact, kl = lf(adapter, ftok, ffield, fz, fmsg, ttok, tfield, tmsg, tref)
            log.append({"step": it, "loss": round(float(loss.item()), 5), "fact_ce": round(float(lfact.item()), 5),
                        "text_kl": round(float(kl.item()), 5), "elapsed_s": round(time.time() - t0, 1)})
            print(log[-1], flush=True)
    adapter.freeze()
    wpath = out / "adapter.safetensors"
    mx.save_safetensors(str(wpath), dict(tree_flatten(adapter.parameters())))
    with wpath.open("rb") as f:
        afp = hashlib.file_digest(f, "sha256").hexdigest()[:12]
    receipt = {"schema": "ens.expA.lm_bus_receipt.v1", "recipe": RECIPE, "base_checkpoint": ckpt,
               "adapter_checkpoint": f"ens-expA-busadapter-qwen3-0.6b-site{args.site}-s{args.seed}@{afp}",
               "adapter": adapter.spec(), "codebook_sha256": C.codebook_digest(args.width), "args": vars(args),
               "dtype": args.dtype, "trainable_params": C.n_params(adapter), "base_params": C.n_params(model),
               "templates": {"train": TEMPLATES_TRAIN, "held": TEMPLATES_HELD}, "attrs": ATTRS,
               "text_corpus": str(TEXT), "log": log, **C.source_revision(), **C.host_record()}
    C.write_json(out / "train_receipt.json", receipt)       # training survives an evaluation crash (review 1)
    return finish(out, receipt, model, adapter, tok, M, args)


def finish(out, receipt, model, adapter, tok, M, args):
    receipt = dict(receipt, eval=evaluate(model, adapter, tok, M, args.seed, args.eval_n))
    C.write_json(out / "receipt.json", receipt)
    print("eval", json.dumps(receipt["eval"]), flush=True)
    return receipt


def eval_only(args):
    """Rerun only the evaluation of a trained adapter (after an evaluation crash)."""
    out = Path(args.out)
    rec = json.loads((out / "train_receipt.json").read_text())
    path = Q.snapshot_dir()
    model, cfg, ckpt = Q.load(path, dtype=mx.bfloat16 if rec["dtype"] == "bf16" else mx.float32)
    if ckpt != rec["base_checkpoint"]:
        raise SystemExit("base checkpoint differs from the training receipt")
    model.freeze()
    s = rec["adapter"]
    adapter = BusAdapter(model, s["site"], s["width"], s["rank"], s["tau"])
    adapter.load_weights(str(out / "adapter.safetensors"))
    with (out / "adapter.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest()[:12] != rec["adapter_checkpoint"].rpartition("@")[2]:
            raise SystemExit("adapter weights differ from the training receipt")
    adapter.freeze()
    return finish(out, rec, model, adapter, Tok(path), C.codebook(s["width"]), args)


CHUNK = 32


def _chunks(n):
    return [(i, min(n, i + CHUNK)) for i in range(0, n, CHUNK)]


def evaluate(model, adapter, tok, M, seed, n):
    """Chunked (review 1). Accuracy over the full vocabulary (top1), among the 64 values, and among the 16 values of
    the asked attribute (the fair chance level is 1/16: the base model already picks the right attribute's kind of
    word; review 3). Wrong messages are reported for the same attribute and for another attribute (review 4). A
    zero-message arm measures what the admitted adapter does with nothing to say."""
    vid = mx.array(tok.value_ids)
    attr_mask = mx.array([[1.0 if z // 16 == a else 0.0 for z in range(N_VALUES)] for a in range(len(ATTR_NAMES))])
    res = {}
    for name, temps in (("train_templates", TEMPLATES_TRAIN), ("held_templates", TEMPLATES_HELD)):
        rng = random.Random(70_000 + seed + len(name))
        fr = fact_rows(tok, rng, n, temps)
        same = [16 * (r[2] // 16) + (r[2] % 16 + 1 + rng.randrange(15)) % 16 for r in fr]
        cross = [(16 * ((r[2] // 16 + 1 + rng.randrange(3)) % 4)) + rng.randrange(16) for r in fr]
        arms = {"base": None, "oracle": "z", "zero_message": "zero", "wrong_same_attr": same, "wrong_cross_attr": cross}
        tally = {}
        for a0, a1 in _chunks(n):
            sub = fr[a0:a1]
            t, f = pad([r[0] for r in sub], tok), mx.array([r[1] for r in sub])
            z = mx.array([r[2] for r in sub])
            am = attr_mask[z // 16]
            for arm, spec in arms.items():
                if spec is None:
                    row = forward(model, t, None, at=f)
                else:
                    msg = M[z] if spec == "z" else (mx.zeros_like(M[z]) if spec == "zero" else M[mx.array(spec[a0:a1])])
                    row = forward(model, t, (f, msg), adapter, at=f)
                row = row.astype(mx.float32)
                vals = row[:, vid]
                pred_top = mx.argmax(row, -1)
                pred_v = mx.argmax(vals, -1)
                pred_a = mx.argmax(vals + (am - 1.0) * 1e9, -1)
                targets = {"true": z}
                if isinstance(spec, list):
                    targets["msg"] = mx.array(spec[a0:a1])
                for tname, tz in targets.items():
                    for k, pred in (("top1", pred_top == vid[tz]), ("among_values", pred_v == tz),
                                    ("among_attr", pred_a == tz)):
                        key = f"{arm}.{tname}.{k}"
                        tally[key] = tally.get(key, 0.0) + float(mx.sum(pred).item())
        res[name] = {"n": n, **{k: round(v / n, 4) for k, v in sorted(tally.items())},
                     "chance_among_values": round(1 / N_VALUES, 4), "chance_among_attr": round(1 / 16, 4)}
    # admitted-but-irrelevant drift on held text, split by position relative to the write (review 5)
    held = text_windows(tok, "held")
    rng = random.Random(80_000 + seed)
    tw = [held[rng.randrange(len(held))] for _ in range(min(n, 256))]
    acc = {k: [0.0, 0] for k in ("kl_before", "kl_at", "kl_after", "kl_zero_msg_all")}
    nll = {"original": [0.0, 0], "admitted": [0.0, 0]}
    for a0, a1 in _chunks(len(tw)):
        tt = mx.array(tw[a0:a1])
        B = tt.shape[0]
        fld = mx.array([rng.randrange(8, 16) for _ in range(B)])
        msg = M[mx.array([rng.randrange(N_VALUES) for _ in range(B)])]
        ref = model(tt).astype(mx.float32)
        adm = forward(model, tt, (fld, msg), adapter).astype(mx.float32)
        zer = forward(model, tt, (fld, mx.zeros_like(msg)), adapter).astype(mx.float32)

        def kl(x):
            return mx.sum(mx.softmax(ref, -1) * (_log_softmax(ref, -1) - _log_softmax(x, -1)), -1)   # (B, T)
        k_adm, k_zero = kl(adm), kl(zer)
        pos = mx.arange(SEQ)[None, :]
        for key, m in (("kl_before", pos < fld[:, None]), ("kl_at", pos == fld[:, None]),
                       ("kl_after", pos > fld[:, None])):
            acc[key][0] += float(mx.sum(k_adm * m).item())
            acc[key][1] += int(mx.sum(m).item())
        acc["kl_zero_msg_all"][0] += float(mx.sum(k_zero).item())
        acc["kl_zero_msg_all"][1] += k_zero.size
        for key, lg in (("original", ref), ("admitted", adm)):
            ce = nn.losses.cross_entropy(lg[:, :-1], tt[:, 1:], reduction="none")
            nll[key][0] += float(mx.sum(ce).item())
            nll[key][1] += ce.size
    res["held_text"] = {"windows": len(tw), "field_positions": "8-15",
                        **{k: round(s / max(1, c), 6) for k, (s, c) in acc.items()},
                        **{f"nll_{k}": round(s / c, 5) for k, (s, c) in nll.items()}}
    res["held_text"]["nll_cost"] = round(res["held_text"]["nll_admitted"] - res["held_text"]["nll_original"], 5)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", type=int, default=8, help="bus after this block (0-based; Qwen3-0.6B has 28)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--reader-init", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--p-text", type=float, default=0.25)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-only", action="store_true", help="rerun the evaluation of a trained adapter in --out")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    eval_only(a) if a.eval_only else train(a)


if __name__ == "__main__":
    main()
