"""H19b / H21a: can an attention tile add depth inside one core call?

The frozen bus-trained core cannot do k-step GRAPH walks or FST runs in one call, even with the oracle message
M[z] (single call 0.31-0.44). H7 adds depth outside the core by chaining calls. Here one call gets a tile that
reads the provided intermediate M[z] and the prompt (the tap view of every position before the field) and runs
r internal attention rounds before writing the bus message. It is trained through the frozen core's answer loss.

  r0        no prompt access: a learned map of M[z] alone (can only re-encode z: the single-call control)
  attn-r    r rounds of ordinary attention over the prompt plus a small learned memory
  tri-r     r rounds of 2-simplicial (triplet) attention over position pairs (H21a)

Rows are generated ad hoc (z random, no hidden law, no sealed family); train and held rows are split by
operation hash, so held rows test new tables and bit strings.

  python -m research.ens.depth --core <c6 dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from research.ens import common as C
from research.ens.chain import _rows
from research.ens.contrib import check_core_families, core_site, unit_rms
from research.ens.model import BusCall, value_logits
from research.ens.route import site_hidden


class _Box:
    z = None


class DepthTile(nn.Module):
    reads_sequence = True

    def __init__(self, w: int, rounds: int, mode: str, n_mem: int = 16, raw_d: int = 0):
        super().__init__()
        self.rounds, self.mode = rounds, mode
        self.reads_raw = raw_d > 0           # read the core's d-wide state instead of the frozen 64-wide tap
        if self.reads_raw:
            self.proj = nn.Linear(raw_d, w)
        self.inp = nn.Linear(w, w)
        self.q = [nn.Linear(w, w, bias=False) for _ in range(rounds)]
        self.k = [nn.Linear(w, w, bias=False) for _ in range(rounds)]
        self.v = [nn.Linear(w, w, bias=False) for _ in range(rounds)]
        if mode == "tri":
            self.k2 = [nn.Linear(w, w, bias=False) for _ in range(rounds)]
            self.v2 = [nn.Linear(w, w, bias=False) for _ in range(rounds)]
        self.o = [nn.Linear(w, w) for _ in range(rounds)]
        self.ff = [nn.Sequential(nn.Linear(w, 4 * w), nn.SiLU(), nn.Linear(4 * w, w)) for _ in range(rounds)]
        self.norm = [nn.RMSNorm(w) for _ in range(rounds)]
        self.mem = mx.random.normal((n_mem, w)) * 0.02
        self.final_norm = nn.RMSNorm(w)       # without it the state grows and unit_rms(out(s)) stops depending
        self.out = nn.Linear(w, w)            # on the input: gradients upstream collapse to ~0 (debug 2026-11-21)
        self.box = _Box()                     # per batch: the provided intermediate M[z] (not a parameter)

    def __call__(self, seq, valid):
        s = self.inp(self.box.z)
        if self.rounds == 0:
            return unit_rms(self.out(self.final_norm(s)))
        if self.reads_raw:
            seq = self.proj(seq) * valid[..., None]
        B, T, w = seq.shape
        mem = mx.broadcast_to(self.mem[None], (B,) + self.mem.shape)
        ctx = mx.concatenate([seq, mem], axis=1)
        ok = mx.concatenate([valid, mx.ones((B, self.mem.shape[0]))], axis=1) > 0
        for r in range(self.rounds):
            q = self.q[r](self.norm[r](s))
            if self.mode == "attn":
                sc = mx.sum(q[:, None, :] * self.k[r](ctx), -1) / w ** 0.5
                sc = mx.where(ok, sc, -1e9)
                a = mx.softmax(sc, -1)
                upd = mx.sum(a[..., None] * self.v[r](ctx), 1)
            else:                                     # 2-simplicial: score(i, j) = <q, k1_i * k2_j>
                k1, k2 = self.k[r](ctx), self.k2[r](ctx)
                sc = mx.einsum("bd,bid,bjd->bij", q, k1, k2) / w ** 0.5
                pair_ok = ok[:, :, None] & ok[:, None, :]
                sc = mx.where(pair_ok, sc, -1e9)
                N = sc.shape[1]
                a = mx.softmax(sc.reshape(B, N * N), -1).reshape(B, N, N)
                v1, v2 = self.v[r](ctx), self.v2[r](ctx)
                upd = mx.einsum("bij,bid,bjd->bd", a, v1, v2)
            s = s + self.o[r](upd)
            s = s + self.ff[r](self.norm[r](s))
        return unit_rms(self.out(self.final_norm(s)))


class _ShufView:
    """Hands the tile another row's prompt view (and validity mask) for the shuffled-prompt control."""
    reads_sequence = True

    def __init__(self, tile, perm):
        self.tile, self.perm = tile, perm
        self.reads_raw = tile.reads_raw

    def __call__(self, seq, valid):
        return self.tile(seq[self.perm], valid[self.perm])


def _split(data, held: bool):
    return [d for d in data if (hashlib.sha256(repr(d[2].op).encode()).digest()[0] < 32) == held]   # 1/8 held


def _parse(arm):
    """'attn2', 'tri1', 'r0', with optional prefixes: 'raw-' (read the core's d-wide state, not the tap),
    then 'alone-' (the tile with its own answer head, no core) or 'shuf-' (the tile reads another row's prompt). Review 2026-11-21: without these, a win cannot be attributed
    to depth inside the call rather than the tile solving the task and steering the core."""
    variant, raw = "core", arm.startswith("raw-")
    if raw:
        arm = arm[4:]
    for pre in ("alone-", "shuf-"):
        if arm.startswith(pre):
            variant, arm = pre[:-1], arm[len(pre):]
    if arm.startswith("attn"):
        return variant, "attn", int(arm[4:]), raw
    if arm.startswith("tri"):
        return variant, "tri", int(arm[3:]), raw
    if arm == "r0":
        return variant, "attn", 0, raw
    raise ValueError(f"unknown depth arm {arm!r}")


def run_cell(core, site, cls, m, k, arm, *, steps=3000, batch=64, lr=1e-3, seed=0) -> dict:
    variant, mode, r, raw = _parse(arm)
    M = C.codebook(core.cfg.bus_width)
    _, _, data = _rows(cls, m, k, 6000, seed)
    train, held = _split(data, False), _split(data, True)
    mx.random.seed(seed)
    tile = DepthTile(core.cfg.bus_width, r, mode, raw_d=core.cfg.d if raw else 0)
    if variant == "alone":
        tile.head = nn.Linear(core.cfg.bus_width, 64)
    opt = optim.Adam(learning_rate=lr)
    perm_rng = random.Random(seed + 99)

    def call(t, tokens, q, f, a):
        if variant == "alone":           # the tile's own head on the same prompt view, no core decode
            port = core.ports[site]
            h = site_hidden(core, tokens, site)           # the same state the core's port taps
            T = h.shape[1]
            before = (mx.arange(T)[None, :] < f[:, None]).astype(h.dtype)[..., None]
            view = h if t.reads_raw else port.tap(h)
            return t.head(t(view * before, before[..., 0]))
        if variant == "shuf":            # same rows, but the tile reads a permuted row's prompt
            B = tokens.shape[0]
            perm = mx.array(perm_rng.sample(range(B), B))
            sv = _ShufView(t, perm)
            return value_logits(core(tokens, BusCall(site, q, f, sv)), a)
        return value_logits(core(tokens, BusCall(site, q, f, t)), a)

    def batch_of(rows):
        tokens, q, f, a, _, _ = C.batch([d[2] for d in rows], "bus")
        return tokens, q, f, a, mx.array([d[3] for d in rows]), M[mx.array([d[0] for d in rows])]

    def lf(t, tokens, q, f, a, ans, zm):
        t.box.z = zm
        return nn.losses.cross_entropy(call(t, tokens, q, f, a), ans, reduction="mean")
    g = nn.value_and_grad(tile, lf)
    rng = random.Random(seed)
    t0 = time.time()
    for _ in range(steps):
        loss, gr = g(tile, *batch_of([train[rng.randrange(len(train))] for _ in range(batch)]))
        opt.update(tile, gr)
        mx.eval(tile.parameters(), opt.state)
    def accuracy(rows):
        tokens, q, f, a, ans, zm = batch_of(rows)
        tile.box.z = zm
        return float(mx.mean(mx.argmax(call(tile, tokens, q, f, a), -1) == ans).item()), (tokens, q, f, a, ans, zm)
    held_acc, (tokens, q, f, a, ans, zm) = accuracy(held)
    train_acc, _ = accuracy(train[:len(held)])
    single = mx.argmax(value_logits(core(tokens, BusCall(site, q, f, zm)), a), -1)
    return {"cls": cls, "m": m, "k": k, "arm": arm, "variant": variant, "n_train": len(train), "n_held": len(held),
            "held_acc": held_acc, "train_acc": train_acc,
            "single_call_oracle": float(mx.mean(single == ans).item()),
            "last_loss": float(loss.item()), "train_s": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cells", nargs="+", default=["FST:3:2", "FST:3:3", "GRAPH:8:2", "GRAPH:8:3"])
    ap.add_argument("--arms", nargs="+", default=["r0", "attn1", "attn2", "attn4", "tri1", "tri2",
                                                  "alone-attn2", "shuf-attn2", "alone-tri1"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    check_core_families(args.core)
    site = core_site(args.core)
    cells = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[2])) for c in args.cells]
    results = []
    for c, m, k in cells:
        for arm in args.arms:
            res = run_cell(core, site, c, m, k, arm, steps=args.steps, lr=args.lr)
            results.append(res)
            print(json.dumps(res), flush=True)
    receipt = {"schema": "ens.a0.depth_receipt.v0", "hypothesis": ["H19b", "H21a"], "core_checkpoint": ckpt,
               "site": site, "results": results, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
