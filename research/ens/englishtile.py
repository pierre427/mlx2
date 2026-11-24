"""H44: an English edge tile on a frozen core (Pierre, 2026-11-23: "freeze seed 1 and build a new fresh English tile").

The core (seed 1 C6, frozen) reads and writes only its closed symbolic vocabulary. The English tile sits at its
edges:

  reader    English bytes -> the core's published symbolic prompt (bus layout, 32 tokens). Trained on parallel
            data: the symbolic prompt grammar is part of the published ABI, so a tile author may use it.
  core      unchanged; the domain tile for the parsed structure writes z at the bus site, as in A0.
  renderer  the core's final state at the answer position -> an English answer sentence (bytes). It reads the
            state, not the argmax token, and is trained to render the core's own answer (faithfulness).

Arms on the same rows (one English tile for all four dev families, one domain tile per family):
  pipeline       English -> reader -> core + domain tile -> renderer
  ceiling        the true symbolic prompt -> core + domain tile (what the core can do)
  no_tile        the reader's prompt -> core with no bus call (does the domain tile matter?)
  english_alone  a byte model of the reader's size trained end to end, English -> answer (does the core add
                 anything?)

Evaluation cells: held rows (operations the domain tiles never trained on) x held wording (frames and phrases the
reader never saw), plus the three other combinations. Kill measures (wiki hypotheses H44): pipeline no better
than english_alone + 0.05 on held x held; held-wording parse far below train-wording parse; renderer unfaithful.

  python -m research.ens.englishtile --core <seed-1 c6 dir> --tiles <dir with mod-<fid>/ per dev family> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import english as E
from research.ens import families as F
from research.ens import vocab as V
from research.ens.contrib import check_core_families, load_module, message
from research.ens.model import BusCall

FAMILIES = ("program.ADD.h1", "graph.6-2.h1", "fst.3-1.h1", "graph.4-2.h1")
ANS_BYTES = 32
RECIPE = "ens.h44.english_tile.v1"


# ------------------------------------------------------------------ models
class Layer(nn.Module):
    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.att = nn.MultiHeadAttention(d, heads)
        if cross:
            self.n2 = nn.RMSNorm(d)
            self.catt = nn.MultiHeadAttention(d, heads)
        self.n3 = nn.RMSNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def __call__(self, x, mask=None, mem=None, mem_mask=None):
        y = self.n1(x)
        x = x + self.att(y, y, y, mask)
        if mem is not None:
            y = self.n2(x)
            x = x + self.catt(y, mem, mem, mem_mask)
        return x + self.ff2(nn.gelu(self.ff1(self.n3(x))))


class Reader(nn.Module):
    """Bytes -> `slots` outputs over `out_vocab` (a byte encoder plus learned slots that cross-attend to it).
    slots=32, out_vocab=128: the English tile's reader. slots=1, out_vocab=64: the english_alone control."""

    def __init__(self, d=192, heads=4, enc_layers=4, dec_layers=2, slots=F.SEQ_LEN, out_vocab=V.VOCAB_SIZE):
        super().__init__()
        self.byte = nn.Embedding(257, d)
        self.pos = nn.Embedding(E.MAX_BYTES, d)
        self.enc = [Layer(d, heads) for _ in range(enc_layers)]
        self.slots = mx.random.normal((slots, d)) * 0.02
        self.dec = [Layer(d, heads, cross=True) for _ in range(dec_layers)]
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, out_vocab)

    def __call__(self, b):
        B, L = b.shape
        pad = mx.where(b == 0, -1e9, 0.0)[:, None, None, :]          # (B, 1, 1, L) additive key mask
        h = self.byte(b) + self.pos(mx.arange(L))
        for lay in self.enc:
            h = lay(h, pad)
        s = mx.broadcast_to(self.slots[None], (B,) + self.slots.shape)
        for lay in self.dec:
            s = lay(s, None, h, pad)
        return self.head(self.norm(s))


class Renderer(nn.Module):
    """Core state (d_core) -> answer bytes, a GRU conditioned on the state through its initial hidden."""

    def __init__(self, d_core=256, hidden=256, emb=64):
        super().__init__()
        self.init = nn.Linear(d_core, hidden)
        self.emb = nn.Embedding(257, emb)
        self.gru = nn.GRU(emb, hidden)
        self.out = nn.Linear(hidden, 257)

    def __call__(self, state, prev):
        return self.out(self.gru(self.emb(prev), mx.tanh(self.init(state))))

    def generate(self, state, n=ANS_BYTES):
        B = state.shape[0]
        h = mx.tanh(self.init(state))
        tok = mx.zeros((B, 1), dtype=mx.int32)                       # 0 = start (also padding / end)
        out = []
        for _ in range(n):
            o = self.gru(self.emb(tok), h)
            h = o[:, -1]
            tok = mx.argmax(self.out(h), axis=-1)[:, None].astype(mx.int32)
            out.append(tok)
        return mx.concatenate(out, axis=1)


# ------------------------------------------------------------------ core with its final state
def core_run(core, tokens, bus):
    """Logits and the final normalized state; bus=None runs the ordinary graph (structural bypass)."""
    if bus is not None:
        core._site_index(bus.site)                     # KeyError for an unknown site, as Core.__call__
    if "lora" in core:
        raise ValueError("core_run does not apply LoRA; H44 runs on a core without it")
    bb = core.backbone
    h = bb.embed_tokens(tokens)
    site_after = {i: name for name, i in core.cfg.sites}
    if bus is not None and site_after.get(-1) == bus.site:
        h = core._apply(h, bus, False)
    for i, blk in enumerate(bb.blocks):
        h = blk(h)
        if bus is not None and site_after.get(i) == bus.site:
            h = core._apply(h, bus, False)
    hn = bb.norm(h)
    return bb.head(hn), hn


# ------------------------------------------------------------------ parsing the reader's prompt
def parse(ids):
    """(family id, query_idx, field_idx, ans_idx) of an emitted prompt, or None when it is not a well-formed bus
    prompt of one of the H44 families (fail closed: a malformed prompt counts as a wrong answer)."""
    ids = [int(i) for i in ids]
    if any(not 0 <= i < len(V.TOKENS) for i in ids):      # the reader's head covers 128 ids, 98 are named
        return None
    if ids[:2] != [V.BOS, V.Q] or not V.K0 <= ids[2] < V.K0 + V.N_KEYS or V.ANS not in ids:
        return None
    a = ids.index(V.ANS)
    if a < 6 or ids[a - 1] != V.NEUTRAL or ids[a - 2] != V.SLOT or any(t != V.PAD for t in ids[a + 1:]):
        return None
    op = ids[3:a - 2]
    if any(t in (V.PAD, V.BOS, V.Q, V.ANS, V.SLOT, V.NEUTRAL) for t in op):
        return None
    head = op[0]
    fid = None
    try:
        if head == V.tok("GR") and V.tok("HOP") in op:
            m = op.index(V.tok("HOP")) - 1
            hops = V.value_of(op[-1])
            if len(op) == m + 3 and all(V.V0 <= t < V.V0 + m for t in op[1:1 + m]):
                fid = f"graph.{m}-{hops}.h1"
        elif head == V.tok("FST") and V.tok("IN") in op and op[-1] == V.tok("END"):
            i = op.index(V.tok("IN"))
            s = (i - 1) // 2
            bits = op[i + 1:-1]
            if (i - 1) % 2 == 0 and all(V.V0 <= t < V.V0 + s for t in op[1:i]) and \
                    all(t in (V.tok("B0"), V.tok("B1")) for t in bits):
                fid = f"fst.{s}-{len(bits)}.h1"
        elif head in (V.tok("ADD"), V.tok("MUL"), V.tok("XOR")) and len(op) % 2 == 0:
            names = [V.TOKENS[op[j]] for j in range(0, len(op), 2)]
            consts = [V.value_of(op[j]) for j in range(1, len(op), 2)]     # ValueError if not a value token
            if all(n in ("ADD", "MUL", "XOR") for n in names) and len(consts) == len(names):
                fid = f"program.{'.'.join(names)}.h1"
    except (ValueError, IndexError):
        return None
    if fid not in FAMILIES:
        return None
    return fid, 2, a - 1, a


# ------------------------------------------------------------------ data
def make_batch(rows_fams, rng, held_wording):
    """rows_fams: [(family, row)]. Returns bytes (B, MAX_BYTES), target prompt ids (B, 32), answers (B,)."""
    b, t, ans = [], [], []
    for fam, r in rows_fams:
        b.append(E.to_bytes(E.phrase(fam, r, rng, held=held_wording)))
        t.append(F.layout(r, "bus")[0])
        ans.append(r.answer)
    return mx.array(b), mx.array(t), mx.array(ans)


AUX_FAMILIES = tuple(f.fid for f in F.split("train") if f.cls == "PROGRAM")
AUX_SHARE = 0.25


def aux_rows(rng, n):
    """Reader-only grammar rows from the core-training PROGRAM families (random key and z, as in core training).
    They give the reader contact with every value token and the MUL / XOR verbs. They carry no dev hidden law and
    no answer target; english_alone (which learns answers) does not use them (review finding 1, 2026-11-23)."""
    out = []
    for _ in range(n):
        fam = F.BY_ID[AUX_FAMILIES[rng.randrange(len(AUX_FAMILIES))]]
        out.append((fam, F.sample_row(fam, rng)))
    return out


def pools(split):
    return {fid: F.family_rows(F.BY_ID[fid], split) for fid in FAMILIES}


def sample(pool, rng, n):
    out = []
    for _ in range(n):
        fid = FAMILIES[rng.randrange(len(FAMILIES))]
        out.append((F.BY_ID[fid], pool[fid][rng.randrange(len(pool[fid]))]))
    return out


def lr_schedule(lr, steps, warmup=500):
    warmup = max(1, min(warmup, steps // 10))
    return optim.join_schedules([optim.linear_schedule(0.0, lr, warmup), optim.cosine_decay(lr, steps - warmup, lr * 0.1)],
                                [warmup])


# ------------------------------------------------------------------ training
def train_reader(model, pool, steps, batch, lr, seed, log_every, *, alone=False):
    opt = optim.AdamW(learning_rate=lr_schedule(lr, steps), weight_decay=0.01)
    rng = random.Random(30_000 + seed + (7 if alone else 0))

    def lf(m, b, t, ans):
        out = m(b)
        if alone:
            return nn.losses.cross_entropy(out[:, 0], ans, reduction="mean")
        return nn.losses.cross_entropy(out, t, reduction="mean")
    g = nn.value_and_grad(model, lf)
    log, t0 = [], time.time()
    for it in range(1, steps + 1):
        n_aux = 0 if alone else int(batch * AUX_SHARE)
        b, t, ans = make_batch(sample(pool, rng, batch - n_aux) + aux_rows(rng, n_aux), rng, held_wording=False)
        loss, gr = g(model, b, t, ans)
        opt.update(model, gr)
        mx.eval(model.parameters(), opt.state)
        if it % log_every == 0 or it == steps:
            log.append({"step": it, "loss": round(float(loss.item()), 5), "elapsed_s": round(time.time() - t0, 1)})
            print(("alone " if alone else "reader ") + json.dumps(log[-1]), flush=True)
    return log


def op_guess_rate(fam, row):
    """Accuracy of the best guess that reads the visible operation but not the hidden law: the share of z values
    mapping to the most common answer (review finding 3)."""
    counts = {}
    for z in range(fam.z_domain):
        v = F.answer_for(fam, row, z)
        counts[v] = counts.get(v, 0) + 1
    return max(counts.values()) / fam.z_domain


def bus_for(tiles, fid, tokens, q, f, site):
    return BusCall(site, q, f, message(tiles[fid], tokens, None, q))


def core_answers(core, tiles, site, tokens, parsed, *, with_tile=True):
    """Core value answers and final states for emitted prompts, grouped by the parsed family (route by structure).
    Rows whose prompt did not parse get answer -1 and a zero state."""
    B = tokens.shape[0]
    answers = [-1] * B
    states = mx.zeros((B, core.cfg.d))
    groups = {}
    for i, p in enumerate(parsed):
        if p is not None:
            groups.setdefault(p[0], []).append(i)
    for fid, idx in groups.items():
        ix = mx.array(idx)
        tok = tokens[ix]
        q = mx.array([parsed[i][1] for i in idx]); f = mx.array([parsed[i][2] for i in idx])
        a = mx.array([parsed[i][3] for i in idx])
        logits, hn = core_run(core, tok, bus_for(tiles, fid, tok, q, f, site) if with_tile else None)
        n = len(idx)
        val = mx.argmax(logits[mx.arange(n), a][:, V.V0:V.V0 + V.N_VALUES], axis=-1)
        st = hn[mx.arange(n), a]
        mx.eval(val, st)
        for j, i in enumerate(idx):
            answers[i] = int(val[j].item())
        states[ix] = st
    return answers, states


def train_renderer(ren, core, tiles, site, pool, steps, batch, lr, seed, log_every):
    """Trained on the core's states for true symbolic prompts (training rows), to say the core's own answer."""
    opt = optim.AdamW(learning_rate=lr_schedule(lr, steps, warmup=200), weight_decay=0.01)
    rng = random.Random(40_000 + seed)

    def lf(m, st, tgt):
        prev = mx.concatenate([mx.zeros((tgt.shape[0], 1), dtype=tgt.dtype), tgt[:, :-1]], axis=1)
        return nn.losses.cross_entropy(m(st, prev), tgt, reduction="mean")
    g = nn.value_and_grad(ren, lf)
    log, t0 = [], time.time()
    for it in range(1, steps + 1):
        rf = sample(pool, rng, batch)
        tokens = mx.array([F.layout(r, "bus")[0] for _, r in rf])
        parsed = [parse(F.layout(r, "bus")[0]) for _, r in rf]
        answers, states = core_answers(core, tiles, site, tokens, parsed)
        tgt = mx.array([E.to_bytes(E.answer_text(a, rng), ANS_BYTES) for a in answers])
        loss, gr = g(ren, states, tgt)
        opt.update(ren, gr)
        mx.eval(ren.parameters(), opt.state)
        if it % log_every == 0 or it == steps:
            log.append({"step": it, "loss": round(float(loss.item()), 5), "elapsed_s": round(time.time() - t0, 1)})
            print("renderer " + json.dumps(log[-1]), flush=True)
    return log


# ------------------------------------------------------------------ evaluation
def evaluate(reader, alone, ren, core, tiles, site, rows_split, held_wording, seed, n_per_family):
    rng = random.Random(50_000 + seed + (1 if held_wording else 0) + (2 if rows_split == "held" else 0))
    pool = pools(rows_split)
    out = {}
    for fid in FAMILIES:
        fam = F.BY_ID[fid]
        rows = pool[fid][:n_per_family]
        rf = [(fam, r) for r in rows]
        b, t, ans = make_batch(rf, rng, held_wording)
        ids = mx.argmax(reader(b), axis=-1)
        parsed = [parse(ids[i].tolist()) for i in range(len(rows))]
        truth = [r.answer for r in rows]
        pipe, states = core_answers(core, tiles, site, ids, parsed)
        notile, _ = core_answers(core, tiles, site, ids, parsed, with_tile=False)
        true_parsed = [parse(F.layout(r, "bus")[0]) for r in rows]
        ceil, _ = core_answers(core, tiles, site, t, true_parsed)
        rendered = [E.read_number(E.from_bytes(x)) if p is not None else None
                    for x, p in zip(ren.generate(states).tolist(), parsed)]
        al = mx.argmax(alone(b)[:, 0], axis=-1).tolist()
        n = len(rows)
        acc = lambda xs: round(sum(int(x == y) for x, y in zip(xs, truth)) / n, 4)
        ok = [p is not None for p in parsed]
        out[fid] = {
            "n": n,
            "parse_exact": round(float(mx.mean(mx.all(ids == t, axis=1)).item()), 4),
            "parse_wellformed": round(sum(ok) / n, 4),
            "pipeline": acc(pipe),
            "pipeline_rendered": acc(rendered),
            # over parsed rows only; 0.0 (and flagged in n_parsed) when nothing parsed
            "renderer_faithful": round(sum(int(r == p) for r, p, k in zip(rendered, pipe, ok) if k) / max(1, sum(ok)), 4),
            "n_parsed": sum(ok),
            "ceiling": acc(ceil),
            "no_tile": acc(notile),
            "english_alone": acc(al),
            "chance": round(1 / fam.z_domain, 4),
            "op_guess": round(sum(op_guess_rate(fam, r) for r in rows) / n, 4),
        }
        if not held_wording and rows_split == "train":
            out[fid]["example"] = E.phrase(fam, rows[0], random.Random(1), held=False)
        if held_wording and rows_split == "held":
            out[fid]["example"] = E.phrase(fam, rows[0], random.Random(1), held=True)
    keys = ("parse_exact", "parse_wellformed", "pipeline", "pipeline_rendered", "renderer_faithful", "ceiling",
            "no_tile", "english_alone", "op_guess")
    mean = {k: round(sum(v[k] for v in out.values()) / len(out), 4) for k in keys}
    res = {"families": out, "mean": mean}
    if rows_split == "train":
        res["note"] = "in-sample rows: the reader and english_alone trained on these rows"
    return res


def kill_measures(results, margin=0.05, parse_gap=0.2, faithful_min=0.95):
    """The three H44 kill measures, computed here so every reader of the receipt applies the same rule."""
    hh = results["rows_held__wording_held"]["mean"]
    ht = results["rows_held__wording_train"]["mean"]
    k = {"core_adds_nothing": {"pipeline": hh["pipeline_rendered"], "english_alone": hh["english_alone"],
                               "margin": margin, "killed": hh["pipeline_rendered"] <= hh["english_alone"] + margin},
         "only_seen_wording": {"parse_held_wording": hh["parse_exact"], "parse_train_wording": ht["parse_exact"],
                               "gap_limit": parse_gap, "killed": ht["parse_exact"] - hh["parse_exact"] > parse_gap},
         "renderer_unfaithful": {"faithful": hh["renderer_faithful"], "min": faithful_min,
                                 "n_parsed": sum(v["n_parsed"] for v in results["rows_held__wording_held"]["families"].values()),
                                 "killed": hh["renderer_faithful"] < faithful_min}}
    k["any_killed"] = any(v["killed"] for v in k.values() if isinstance(v, dict))
    return k


# ------------------------------------------------------------------ main
def run(args) -> dict:
    out = C.fresh_out(args.out)
    mx.random.seed(args.seed)
    core, core_ckpt = C.load_core(args.core)
    core.freeze()
    if core.cfg.loop_body:     # an H45 loop core needs passes=; this trainer would silently run it at one pass
        raise SystemExit("this trainer does not drive the loop port; refusing an H45 loop core")
    check_core_families(args.core)
    site = json.loads((Path(args.core) / "receipt.json").read_text()).get("site", "middle")
    tiles, tile_ckpts = {}, {}
    for fid in FAMILIES:
        d = Path(args.tiles) / f"mod-{fid}"
        rec = json.loads((d / "receipt.json").read_text())
        if rec["core_checkpoint"] != core_ckpt:
            raise SystemExit(f"{d}: tile trained on {rec['core_checkpoint']}, not {core_ckpt}")
        if rec["family"]["fid"] != fid or rec["family_digest"] != F.BY_ID[fid].digest or rec["site"] != site:
            raise SystemExit(f"{d}: tile family, family digest or site differs from today's {fid} at {site}")
        tiles[fid] = load_module(d, core.cfg.bus_width)
        tile_ckpts[fid] = {"dir": d.name, "package_digest": rec["package_digest"], "family_digest": rec["family_digest"]}
    train_pool = pools("train")
    reader = Reader(d=args.d, slots=F.SEQ_LEN, out_vocab=V.VOCAB_SIZE)
    alone = Reader(d=args.d, slots=1, out_vocab=V.N_VALUES)
    ren = Renderer(d_core=core.cfg.d)
    logs = {"reader": train_reader(reader, train_pool, args.steps, args.batch, args.lr, args.seed, args.log_every),
            "english_alone": train_reader(alone, train_pool, args.steps, args.batch, args.lr, args.seed, args.log_every,
                                          alone=True),
            "renderer": train_renderer(ren, core, tiles, site, train_pool, args.render_steps, args.batch, 1e-3,
                                       args.seed, args.log_every)}
    for name, m in (("reader", reader), ("english_alone", alone), ("renderer", ren)):
        m.freeze()
        mx.save_safetensors(str(out / f"{name}.safetensors"), dict(tree_flatten(m.parameters())))
    results = {}
    for rows_split in ("held", "train"):
        for held_wording in (True, False):
            key = f"rows_{rows_split}__wording_{'held' if held_wording else 'train'}"
            results[key] = evaluate(reader, alone, ren, core, tiles, site, rows_split, held_wording, args.seed,
                                    args.eval_n)
            print(key, json.dumps(results[key]["mean"]), flush=True)
    kills = kill_measures(results)
    print("kill_measures", json.dumps(kills), flush=True)
    n_reader = C.n_params(reader)
    receipt = {"schema": "ens.h44.english_tile_receipt.v1", "recipe": RECIPE, "core_checkpoint": core_ckpt,
               "site": site, "tiles": tile_ckpts, "families": {f: F.BY_ID[f].digest for f in FAMILIES},
               "seed": args.seed,
               "budget": {"reader_steps": args.steps, "render_steps": args.render_steps, "batch": args.batch,
                          "lr": args.lr, "d": args.d, "eval_n_per_family": args.eval_n},
               "params": {"reader": n_reader, "english_alone": C.n_params(alone), "renderer": C.n_params(ren)},
               "aux_grammar": {"families": list(AUX_FAMILIES), "share": AUX_SHARE},
               "logs": logs, "results": results, "kill_measures": kills, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", required=True)
    p.add_argument("--tiles", required=True, help="dir holding mod-<fid>/ MLP tiles for the four dev families")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--render-steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--log-every", type=int, default=500)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
